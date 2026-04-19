#include "bridges/ofxStableDiffusionHoloscanBridge.h"

#include "ofxStableDiffusion.h"

#include <chrono>
#include <deque>
#include <future>
#include <mutex>
#include <thread>

#if !defined(_WIN32) && defined(__has_include)
#  if __has_include(<holoscan/holoscan.hpp>)
#    include <holoscan/holoscan.hpp>
#    define OFXSD_HAS_HOLOSCAN 1
#  else
#    define OFXSD_HAS_HOLOSCAN 0
#  endif
#else
#  define OFXSD_HAS_HOLOSCAN 0
#endif

namespace {

using namespace std::chrono_literals;

bool copyPixelsToPreview(
	const ofxStableDiffusionImageFrame& image,
	ofTexture* texture,
	ofxStableDiffusionHoloscanPreviewFrame* preview) {
	if (!texture || !preview || !image.isAllocated()) {
		return false;
	}
	preview->valid = true;
	preview->pixels = image.pixels;
	texture->loadData(preview->pixels);
	return true;
}

ofxStableDiffusionImageRequest makeBridgeRequest(
	const ofxStableDiffusionHoloscanConditioningPacket& packet) {
	ofxStableDiffusionImageRequest request;
	request.mode = ofxStableDiffusionImageMode::ImageToImage;
	request.prompt = packet.prompt;
	request.negativePrompt = packet.negativePrompt;
	request.strength = packet.strength;
	request.width = static_cast<int>(packet.initImage->getWidth());
	request.height = static_cast<int>(packet.initImage->getHeight());
	request.initImage = {
		static_cast<uint32_t>(packet.initImage->getWidth()),
		static_cast<uint32_t>(packet.initImage->getHeight()),
		static_cast<uint32_t>(packet.initImage->getNumChannels()),
		packet.initImage->getData()
	};
	return request;
}

void clearPreview(ofTexture* texture, ofxStableDiffusionHoloscanPreviewFrame* preview) {
	if (texture) {
		texture->clear();
	}
	if (preview) {
		*preview = {};
	}
}

#if OFXSD_HAS_HOLOSCAN

struct HoloscanSharedRuntimeState {
	mutable std::mutex mutex;
	std::deque<ofxStableDiffusionHoloscanFramePacket> pendingFrames;
	std::vector<ofxStableDiffusionImageFrame> finishedImages;
	ofxStableDiffusionHoloscanPreviewFrame previewFrame;
	bool previewDirty = false;
	std::string latestPrompt;
	std::string latestNegativePrompt;
	std::string lastError;
	bool stopRequested = false;
};

void setHoloscanRuntimeError(
	const std::shared_ptr<HoloscanSharedRuntimeState>& state,
	const std::string& error) {
	if (!state) {
		return;
	}
	std::lock_guard<std::mutex> lock(state->mutex);
	state->lastError = error;
}

namespace holoscanops {

class FrameSourceOp : public holoscan::Operator {
public:
	HOLOSCAN_OPERATOR_FORWARD_ARGS(FrameSourceOp)

	FrameSourceOp() = default;

	std::shared_ptr<HoloscanSharedRuntimeState> sharedState;

	void setup(holoscan::OperatorSpec& spec) override {
		spec.output<std::shared_ptr<ofxStableDiffusionHoloscanFramePacket>>("out");
		spec.param(allocator_,
			"allocator",
			"Allocator",
			"Allocator resource used by the Holoscan frame source.",
			{});
	}

	void compute(
		holoscan::InputContext&,
		holoscan::OutputContext& op_output,
		holoscan::ExecutionContext&) override {
		if (!sharedState) {
			return;
		}

		ofxStableDiffusionHoloscanFramePacket packet;
		{
			std::lock_guard<std::mutex> lock(sharedState->mutex);
			if (sharedState->stopRequested || sharedState->pendingFrames.empty()) {
				return;
			}
			packet = sharedState->pendingFrames.front();
			sharedState->pendingFrames.pop_front();
		}

		if (!packet.isValid()) {
			return;
		}

		op_output.emit(
			std::make_shared<ofxStableDiffusionHoloscanFramePacket>(std::move(packet)),
			"out");
	}

private:
	holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

class ConditioningOp : public holoscan::Operator {
public:
	HOLOSCAN_OPERATOR_FORWARD_ARGS(ConditioningOp)

	ConditioningOp() = default;

	std::shared_ptr<HoloscanSharedRuntimeState> sharedState;

	void setup(holoscan::OperatorSpec& spec) override {
		spec.input<std::shared_ptr<ofxStableDiffusionHoloscanFramePacket>>("in");
		spec.output<std::shared_ptr<ofxStableDiffusionHoloscanConditioningPacket>>("out");
		spec.param(allocator_,
			"allocator",
			"Allocator",
			"Allocator resource used by the Holoscan conditioning stage.",
			{});
	}

	void compute(
		holoscan::InputContext& op_input,
		holoscan::OutputContext& op_output,
		holoscan::ExecutionContext&) override {
		auto packet =
			op_input.receive<std::shared_ptr<ofxStableDiffusionHoloscanFramePacket>>("in");
		if (!packet) {
			return;
		}
		auto framePacket = packet.value();
		if (!framePacket || !framePacket->isValid()) {
			return;
		}

		ofxStableDiffusionHoloscanConditioningPacket conditioning;
		conditioning.frameIndex = framePacket->frameIndex;
		conditioning.timestampSeconds = framePacket->timestampSeconds;
		conditioning.initImage = framePacket->pixels;

		{
			std::lock_guard<std::mutex> lock(sharedState->mutex);
			conditioning.prompt = sharedState->latestPrompt;
			conditioning.negativePrompt = sharedState->latestNegativePrompt;
		}

		if (!conditioning.isValid()) {
			setHoloscanRuntimeError(
				sharedState,
				"Holoscan bridge needs a prompt before it can submit a frame.");
			return;
		}

		op_output.emit(
			std::make_shared<ofxStableDiffusionHoloscanConditioningPacket>(
				std::move(conditioning)),
			"out");
	}

private:
	holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

class DiffusionOp : public holoscan::Operator {
public:
	HOLOSCAN_OPERATOR_FORWARD_ARGS(DiffusionOp)

	DiffusionOp() = default;

	ofxStableDiffusion* diffusion = nullptr;
	std::shared_ptr<HoloscanSharedRuntimeState> sharedState;

	void setup(holoscan::OperatorSpec& spec) override {
		spec.input<std::shared_ptr<ofxStableDiffusionHoloscanConditioningPacket>>("in");
		spec.output<std::shared_ptr<ofxStableDiffusionHoloscanImagePacket>>("out");
		spec.param(allocator_,
			"allocator",
			"Allocator",
			"Allocator resource used by the Holoscan diffusion stage.",
			{});
	}

	void compute(
		holoscan::InputContext& op_input,
		holoscan::OutputContext& op_output,
		holoscan::ExecutionContext&) override {
		auto packet =
			op_input.receive<std::shared_ptr<ofxStableDiffusionHoloscanConditioningPacket>>(
				"in");
		if (!packet) {
			return;
		}
		auto conditioning = packet.value();
		if (!conditioning || !conditioning->isValid() || !diffusion) {
			return;
		}

		std::lock_guard<std::mutex> lock(diffusionMutex_);
		try {
			diffusion->generate(makeBridgeRequest(*conditioning));
			while (diffusion->isGenerating()) {
				std::this_thread::sleep_for(5ms);
			}
			const auto result = diffusion->getLastResult();
			if (!result.success) {
				setHoloscanRuntimeError(sharedState, result.error);
				return;
			}
			if (result.images.empty()) {
				setHoloscanRuntimeError(
					sharedState,
					"Holoscan diffusion stage completed without returning any images.");
				return;
			}

			ofxStableDiffusionHoloscanImagePacket imagePacket;
			imagePacket.frameIndex = conditioning->frameIndex;
			imagePacket.timestampSeconds = conditioning->timestampSeconds;
			imagePacket.imageFrame = result.images.front();

			op_output.emit(
				std::make_shared<ofxStableDiffusionHoloscanImagePacket>(
					std::move(imagePacket)),
				"out");
		} catch (const std::exception& e) {
			setHoloscanRuntimeError(sharedState, e.what());
		}
	}

private:
	holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
	std::mutex diffusionMutex_;
};

class PreviewSinkOp : public holoscan::Operator {
public:
	HOLOSCAN_OPERATOR_FORWARD_ARGS(PreviewSinkOp)

	PreviewSinkOp() = default;

	std::shared_ptr<HoloscanSharedRuntimeState> sharedState;

	void setup(holoscan::OperatorSpec& spec) override {
		spec.input<std::shared_ptr<ofxStableDiffusionHoloscanImagePacket>>("in");
		spec.param(allocator_,
			"allocator",
			"Allocator",
			"Allocator resource used by the Holoscan preview sink.",
			{});
	}

	void compute(
		holoscan::InputContext& op_input,
		holoscan::OutputContext&,
		holoscan::ExecutionContext&) override {
		auto packet =
			op_input.receive<std::shared_ptr<ofxStableDiffusionHoloscanImagePacket>>("in");
		if (!packet) {
			return;
		}
		auto imagePacket = packet.value();
		if (!imagePacket || !imagePacket->imageFrame.isAllocated() || !sharedState) {
			return;
		}

		std::lock_guard<std::mutex> lock(sharedState->mutex);
		sharedState->previewFrame.valid = true;
		sharedState->previewFrame.frameIndex = imagePacket->frameIndex;
		sharedState->previewFrame.timestampSeconds = imagePacket->timestampSeconds;
		sharedState->previewFrame.pixels = imagePacket->imageFrame.pixels;
		sharedState->previewDirty = true;
		sharedState->finishedImages.push_back(imagePacket->imageFrame);
	}

private:
	holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

} // namespace holoscanops

class HoloscanImagePipelineApp : public holoscan::Application {
public:
	HoloscanImagePipelineApp(
		ofxStableDiffusion* diffusion,
		std::shared_ptr<HoloscanSharedRuntimeState> sharedState,
		ofxStableDiffusionHoloscanSettings settings)
		: diffusion_(diffusion)
		, sharedState_(std::move(sharedState))
		, settings_(settings) {
	}

	void compose() override {
		using holoscan::Arg;

		auto allocator = make_resource<holoscan::UnboundedAllocator>("bridge_allocator");

		std::shared_ptr<holoscan::Scheduler> schedulerResource;
		if (settings_.useEventScheduler) {
			schedulerResource = make_scheduler<holoscan::EventBasedScheduler>(
				"event_scheduler",
				Arg("worker_thread_number",
					static_cast<int64_t>(std::max(1, settings_.workerThreads))),
				Arg("stop_on_deadlock", true));
		} else {
			schedulerResource = make_scheduler<holoscan::GreedyScheduler>(
				"greedy_scheduler",
				Arg("stop_on_deadlock", true));
		}
		scheduler(schedulerResource);

		auto frameSource = make_operator<holoscanops::FrameSourceOp>(
			"frame_source",
			Arg("allocator", allocator));
		auto conditioning = make_operator<holoscanops::ConditioningOp>(
			"conditioning",
			Arg("allocator", allocator));
		auto diffusion = make_operator<holoscanops::DiffusionOp>(
			"diffusion",
			Arg("allocator", allocator));
		auto previewSink = make_operator<holoscanops::PreviewSinkOp>(
			"preview_sink",
			Arg("allocator", allocator));

		frameSource->sharedState = sharedState_;
		conditioning->sharedState = sharedState_;
		diffusion->sharedState = sharedState_;
		diffusion->diffusion = diffusion_;
		previewSink->sharedState = sharedState_;

		add_flow(frameSource, conditioning, {{"out", "in"}});
		add_flow(conditioning, diffusion, {{"out", "in"}});
		add_flow(diffusion, previewSink, {{"out", "in"}});
	}

private:
	ofxStableDiffusion* diffusion_ = nullptr;
	std::shared_ptr<HoloscanSharedRuntimeState> sharedState_;
	ofxStableDiffusionHoloscanSettings settings_;
};

#endif // OFXSD_HAS_HOLOSCAN

} // namespace

struct ofxStableDiffusionHoloscanBridge::Impl {
	ofxStableDiffusion* diffusion = nullptr;
	ofxStableDiffusionHoloscanSettings settings;
	bool configured = false;
	bool running = false;
	bool holoscanAvailable = false;
	bool usingHoloscanRuntime = false;
	std::string lastError;
	std::string latestPrompt;
	std::string latestNegativePrompt;
	uint64_t nextFrameIndex = 1;
	ofTexture previewTexture;
	ofxStableDiffusionHoloscanPreviewFrame previewFrame;
	std::deque<ofxStableDiffusionHoloscanFramePacket> pendingFrames;
	std::vector<ofxStableDiffusionImageFrame> finishedImages;
	std::mutex mutex;
	bool requestInFlight = false;
	uint64_t inFlightFrameIndex = 0;
	double inFlightTimestampSeconds = 0.0;

#if OFXSD_HAS_HOLOSCAN
	std::shared_ptr<HoloscanSharedRuntimeState> holoscanState;
	std::shared_ptr<HoloscanImagePipelineApp> holoscanApp;
	std::future<void> holoscanFuture;
#endif

	void clearFallbackState() {
		std::lock_guard<std::mutex> lock(mutex);
		pendingFrames.clear();
		finishedImages.clear();
		requestInFlight = false;
		inFlightFrameIndex = 0;
		inFlightTimestampSeconds = 0.0;
	}

	void clearAllPreviewState() {
		clearPreview(&previewTexture, &previewFrame);
	}
};

ofxStableDiffusionHoloscanBridge::ofxStableDiffusionHoloscanBridge()
	: impl_(std::make_unique<Impl>()) {
#if OFXSD_HAS_HOLOSCAN
	impl_->holoscanAvailable = true;
	impl_->holoscanState = std::make_shared<HoloscanSharedRuntimeState>();
#endif
}

ofxStableDiffusionHoloscanBridge::~ofxStableDiffusionHoloscanBridge() = default;

bool ofxStableDiffusionHoloscanBridge::setup(
	ofxStableDiffusion* diffusion,
	const ofxStableDiffusionHoloscanSettings& settings) {
	impl_->diffusion = diffusion;
	impl_->settings = settings;
	impl_->configured = (diffusion != nullptr);
	impl_->lastError.clear();
	if (!impl_->configured) {
		impl_->lastError = "Holoscan bridge setup requires a valid ofxStableDiffusion instance.";
	}
#if OFXSD_HAS_HOLOSCAN
	if (impl_->holoscanState) {
		std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
		impl_->holoscanState->latestPrompt = impl_->latestPrompt;
		impl_->holoscanState->latestNegativePrompt = impl_->latestNegativePrompt;
		impl_->holoscanState->lastError.clear();
		impl_->holoscanState->stopRequested = false;
	}
#endif
	return impl_->configured;
}

void ofxStableDiffusionHoloscanBridge::shutdown() {
	stop();
	impl_->configured = false;
	impl_->diffusion = nullptr;
	impl_->lastError.clear();
}

bool ofxStableDiffusionHoloscanBridge::startImagePipeline() {
	if (!impl_->configured || impl_->diffusion == nullptr) {
		impl_->lastError = "Holoscan bridge is not configured.";
		return false;
	}

	impl_->lastError.clear();
	impl_->clearFallbackState();
	impl_->clearAllPreviewState();

#if OFXSD_HAS_HOLOSCAN
	if (impl_->settings.enabled && impl_->holoscanAvailable && impl_->holoscanState) {
		{
			std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
			impl_->holoscanState->pendingFrames.clear();
			impl_->holoscanState->finishedImages.clear();
			impl_->holoscanState->previewFrame = {};
			impl_->holoscanState->previewDirty = false;
			impl_->holoscanState->lastError.clear();
			impl_->holoscanState->stopRequested = false;
			impl_->holoscanState->latestPrompt = impl_->latestPrompt;
			impl_->holoscanState->latestNegativePrompt = impl_->latestNegativePrompt;
		}

		impl_->holoscanApp = std::make_shared<HoloscanImagePipelineApp>(
			impl_->diffusion,
			impl_->holoscanState,
			impl_->settings);

		try {
			impl_->holoscanFuture = impl_->holoscanApp->run_async();
			impl_->usingHoloscanRuntime = true;
			impl_->running = true;
			return true;
		} catch (const std::exception& e) {
			impl_->lastError = e.what();
			impl_->holoscanApp.reset();
			impl_->usingHoloscanRuntime = false;
		}
	}
#endif

	impl_->usingHoloscanRuntime = false;
	impl_->running = true;
	return true;
}

void ofxStableDiffusionHoloscanBridge::stop() {
	if (!impl_->running && !impl_->usingHoloscanRuntime) {
		impl_->clearFallbackState();
		impl_->clearAllPreviewState();
		return;
	}

#if OFXSD_HAS_HOLOSCAN
	if (impl_->usingHoloscanRuntime) {
		if (impl_->holoscanState) {
			std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
			impl_->holoscanState->stopRequested = true;
			impl_->holoscanState->pendingFrames.clear();
		}
		if (impl_->holoscanApp) {
			try {
				impl_->holoscanApp->stop_execution();
			} catch (...) {
			}
		}
		if (impl_->holoscanFuture.valid()) {
			try {
				impl_->holoscanFuture.get();
			} catch (const std::exception& e) {
				impl_->lastError = e.what();
			}
		}
		impl_->holoscanApp.reset();
		impl_->usingHoloscanRuntime = false;
		if (impl_->holoscanState) {
			std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
			impl_->holoscanState->finishedImages.clear();
			impl_->holoscanState->previewFrame = {};
			impl_->holoscanState->previewDirty = false;
		}
	}
#endif

	impl_->running = false;
	impl_->clearFallbackState();
	impl_->clearAllPreviewState();
}

void ofxStableDiffusionHoloscanBridge::update() {
	if (!impl_->running || !impl_->diffusion) {
		return;
	}

#if OFXSD_HAS_HOLOSCAN
	if (impl_->usingHoloscanRuntime && impl_->holoscanState) {
		std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
		impl_->lastError = impl_->holoscanState->lastError;
		if (impl_->holoscanState->previewDirty && impl_->holoscanState->previewFrame.valid) {
			impl_->previewFrame = impl_->holoscanState->previewFrame;
			impl_->previewTexture.loadData(impl_->previewFrame.pixels);
			impl_->holoscanState->previewDirty = false;
		}
		return;
	}
#endif

	if (impl_->requestInFlight) {
		if (!impl_->diffusion->isGenerating() && impl_->diffusion->isDiffused()) {
			const auto result = impl_->diffusion->getLastResult();
			if (result.success && !result.images.empty()) {
				auto imageResult = result.images.front();
				copyPixelsToPreview(imageResult, &impl_->previewTexture, &impl_->previewFrame);
				impl_->previewFrame.frameIndex = impl_->inFlightFrameIndex;
				impl_->previewFrame.timestampSeconds = impl_->inFlightTimestampSeconds;
				std::lock_guard<std::mutex> lock(impl_->mutex);
				impl_->finishedImages.push_back(std::move(imageResult));
			} else if (!result.success) {
				impl_->lastError = result.error;
			}
			impl_->requestInFlight = false;
			impl_->inFlightFrameIndex = 0;
			impl_->inFlightTimestampSeconds = 0.0;
		}
		return;
	}

	if (impl_->diffusion->isGenerating()) {
		return;
	}

	ofxStableDiffusionHoloscanFramePacket frame;
	{
		std::lock_guard<std::mutex> lock(impl_->mutex);
		if (impl_->pendingFrames.empty()) {
			return;
		}
		frame = impl_->pendingFrames.front();
		impl_->pendingFrames.pop_front();
	}

	if (!frame.isValid()) {
		return;
	}
	if (impl_->latestPrompt.empty()) {
		impl_->lastError = "Holoscan bridge needs a prompt before it can submit a frame.";
		return;
	}

	ofxStableDiffusionHoloscanConditioningPacket conditioning;
	conditioning.frameIndex = frame.frameIndex;
	conditioning.timestampSeconds = frame.timestampSeconds;
	conditioning.prompt = impl_->latestPrompt;
	conditioning.negativePrompt = impl_->latestNegativePrompt;
	conditioning.initImage = frame.pixels;

	impl_->diffusion->generate(makeBridgeRequest(conditioning));
	impl_->requestInFlight = true;
	impl_->inFlightFrameIndex = conditioning.frameIndex;
	impl_->inFlightTimestampSeconds = conditioning.timestampSeconds;
}

void ofxStableDiffusionHoloscanBridge::submitFrame(
	const ofPixels& pixels,
	double timestampSeconds,
	const std::string& sourceLabel) {
	if (!pixels.isAllocated()) {
		impl_->lastError = "Holoscan bridge received an empty frame.";
		return;
	}
	auto ownedPixels = std::make_shared<ofPixels>(pixels);
	ofxStableDiffusionHoloscanFramePacket packet;
	packet.frameIndex = impl_->nextFrameIndex++;
	packet.timestampSeconds = timestampSeconds;
	packet.pixels = ownedPixels;
	packet.sourceLabel = sourceLabel;

#if OFXSD_HAS_HOLOSCAN
	if (impl_->usingHoloscanRuntime && impl_->holoscanState) {
		std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
		impl_->holoscanState->pendingFrames.push_back(std::move(packet));
		return;
	}
#endif

	std::lock_guard<std::mutex> lock(impl_->mutex);
	impl_->pendingFrames.push_back(std::move(packet));
}

void ofxStableDiffusionHoloscanBridge::submitPrompt(
	const std::string& prompt,
	const std::string& negativePrompt) {
	impl_->latestPrompt = prompt;
	impl_->latestNegativePrompt = negativePrompt;
#if OFXSD_HAS_HOLOSCAN
	if (impl_->holoscanState) {
		std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
		impl_->holoscanState->latestPrompt = prompt;
		impl_->holoscanState->latestNegativePrompt = negativePrompt;
	}
#endif
}

bool ofxStableDiffusionHoloscanBridge::hasPreviewFrame() const {
	return impl_->previewFrame.valid && impl_->previewTexture.isAllocated();
}

const ofTexture& ofxStableDiffusionHoloscanBridge::getPreviewTexture() const {
	return impl_->previewTexture;
}

ofxStableDiffusionHoloscanPreviewFrame
ofxStableDiffusionHoloscanBridge::getPreviewFrameCopy() const {
	return impl_->previewFrame;
}

std::vector<ofxStableDiffusionImageFrame>
ofxStableDiffusionHoloscanBridge::consumeFinishedImages() {
#if OFXSD_HAS_HOLOSCAN
	if (impl_->usingHoloscanRuntime && impl_->holoscanState) {
		std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
		auto images = std::move(impl_->holoscanState->finishedImages);
		impl_->holoscanState->finishedImages.clear();
		return images;
	}
#endif

	std::lock_guard<std::mutex> lock(impl_->mutex);
	auto images = std::move(impl_->finishedImages);
	impl_->finishedImages.clear();
	return images;
}

bool ofxStableDiffusionHoloscanBridge::isConfigured() const {
	return impl_->configured;
}

bool ofxStableDiffusionHoloscanBridge::isRunning() const {
	return impl_->running;
}

bool ofxStableDiffusionHoloscanBridge::isHoloscanAvailable() const {
	return impl_->holoscanAvailable;
}

std::string ofxStableDiffusionHoloscanBridge::getLastError() const {
#if OFXSD_HAS_HOLOSCAN
	if (impl_->usingHoloscanRuntime && impl_->holoscanState) {
		std::lock_guard<std::mutex> lock(impl_->holoscanState->mutex);
		if (!impl_->holoscanState->lastError.empty()) {
			return impl_->holoscanState->lastError;
		}
	}
#endif
	return impl_->lastError;
}

const ofxStableDiffusionHoloscanSettings&
ofxStableDiffusionHoloscanBridge::getSettings() const {
	return impl_->settings;
}
