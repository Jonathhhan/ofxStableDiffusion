#pragma once

#include "../ofxStableDiffusionThread.h"

#include <string>
#include <cmath>
#include <sstream>
#include <utility>
#include <vector>

namespace ofxStableDiffusionNativeAdapter {

inline const char* emptyToNull(const std::string& value) {
	return value.empty() ? nullptr : value.c_str();
}

inline std::string quoteCliArg(const std::string& value) {
	std::string escaped = "\"";
	for (char c : value) {
		if (c == '"') {
			escaped += "\\\"";
		} else {
			escaped += c;
		}
	}
	escaped += "\"";
	return escaped;
}

inline std::string formatCliFloat(float value) {
	std::ostringstream stream;
	stream.setf(std::ios::fixed);
	stream.precision(3);
	stream << value;
	std::string formatted = stream.str();
	while (!formatted.empty() && formatted.back() == '0') {
		formatted.pop_back();
	}
	if (!formatted.empty() && formatted.back() == '.') {
		formatted.pop_back();
	}
	return formatted.empty() ? "0" : formatted;
}

inline const char* sampleMethodToCliName(sample_method_t method) {
	switch (method) {
	case EULER_A_SAMPLE_METHOD: return "euler_a";
	case EULER_SAMPLE_METHOD: return "euler";
	case HEUN_SAMPLE_METHOD: return "heun";
	case DPM2_SAMPLE_METHOD: return "dpm2";
	case DPMPP2S_A_SAMPLE_METHOD: return "dpm++2s_a";
	case DPMPP2M_SAMPLE_METHOD: return "dpm++2m";
	case DPMPP2Mv2_SAMPLE_METHOD: return "dpm++2mv2";
	case LCM_SAMPLE_METHOD: return "lcm";
	default:
		return "euler";
	}
}

inline const char* schedulerToCliName(scheduler_t scheduler) {
	switch (scheduler) {
	case DISCRETE_SCHEDULER: return "discrete";
	case KARRAS_SCHEDULER: return "karras";
	case EXPONENTIAL_SCHEDULER: return "exponential";
	case AYS_SCHEDULER: return "ays";
	case GITS_SCHEDULER: return "gits";
	case SGM_UNIFORM_SCHEDULER: return "sgm_uniform";
	case SIMPLE_SCHEDULER: return "simple";
	case SMOOTHSTEP_SCHEDULER: return "smoothstep";
	case KL_OPTIMAL_SCHEDULER: return "kl_optimal";
	case LCM_SCHEDULER: return "lcm";
	case BONG_TANGENT_SCHEDULER: return "bong_tangent";
	case SCHEDULER_COUNT:
	default:
		return "discrete";
	}
}

inline sample_method_t resolveSampleMethod(sd_ctx_t* sdCtx, sample_method_t requested) {
	if (requested == SAMPLE_METHOD_COUNT) {
		return sd_get_default_sample_method(sdCtx);
	}
	return requested;
}

inline scheduler_t resolveScheduler(
	sd_ctx_t* sdCtx,
	sample_method_t sampleMethod,
	scheduler_t requestedSchedule) {
	if (requestedSchedule == SCHEDULER_COUNT) {
		return sd_get_default_scheduler(sdCtx, sampleMethod);
	}
	return requestedSchedule;
}

inline sd_tiling_params_t makeTilingParams(bool enabled) {
	sd_tiling_params_t tiling{};
	tiling.enabled = enabled;
	tiling.target_overlap = 0.5f;
	return tiling;
}

inline std::string buildEffectivePrompt(const ofxStableDiffusionImageRequest& request) {
	if (request.instruction.empty()) {
		return request.prompt;
	}
	if (request.prompt.empty()) {
		return request.instruction;
	}
	if (request.instruction == request.prompt) {
		return request.prompt;
	}
	return request.prompt + "\nInstruction: " + request.instruction;
}

inline bool looksLikeEmbeddingFile(const std::string& extension) {
	return extension == "pt" ||
		extension == "ckpt" ||
		extension == "safetensors" ||
		extension == "bin" ||
		extension == "gguf";
}

inline void collectEmbeddings(
	const std::string& embedDir,
	std::vector<std::string>& embeddingNames,
	std::vector<std::string>& embeddingPaths,
	std::vector<sd_embedding_t>& embeddings) {
	embeddingNames.clear();
	embeddingPaths.clear();
	embeddings.clear();

	if (embedDir.empty()) {
		return;
	}

	ofDirectory directory(embedDir);
	if (!directory.exists()) {
		return;
	}

	directory.listDir();
	for (std::size_t i = 0; i < directory.size(); ++i) {
		const ofFile& file = directory.getFile(static_cast<int>(i));
		if (!file.isFile()) {
			continue;
		}
		const std::string extension = ofToLower(file.getExtension());
		if (!looksLikeEmbeddingFile(extension)) {
			continue;
		}
		embeddingPaths.push_back(file.getAbsolutePath());
		embeddingNames.push_back(file.getBaseName());
	}

	embeddings.reserve(embeddingPaths.size());
	for (std::size_t i = 0; i < embeddingPaths.size(); ++i) {
		embeddings.push_back({embeddingNames[i].c_str(), embeddingPaths[i].c_str()});
	}
}

inline bool looksLikeImageFile(const std::string& extension) {
	return extension == "png" ||
		extension == "jpg" ||
		extension == "jpeg" ||
		extension == "bmp" ||
		extension == "webp";
}

inline void collectPhotoMakerImages(
	const std::string& imageDir,
	std::vector<ofPixels>& loadedPixels,
	std::vector<sd_image_t>& imageViews) {
	loadedPixels.clear();
	imageViews.clear();

	if (imageDir.empty()) {
		return;
	}

	ofDirectory directory(imageDir);
	if (!directory.exists()) {
		return;
	}

	directory.listDir();
	for (std::size_t i = 0; i < directory.size(); ++i) {
		const ofFile& file = directory.getFile(static_cast<int>(i));
		if (!file.isFile()) {
			continue;
		}
		const std::string extension = ofToLower(file.getExtension());
		if (!looksLikeImageFile(extension)) {
			continue;
		}

		ofPixels pixels;
		if (!ofLoadImage(pixels, file.getAbsolutePath())) {
			continue;
		}

		loadedPixels.push_back(std::move(pixels));
	}

	imageViews.reserve(loadedPixels.size());
	for (auto& pixels : loadedPixels) {
		if (!pixels.isAllocated()) {
			continue;
		}
		imageViews.push_back({
			static_cast<uint32_t>(pixels.getWidth()),
			static_cast<uint32_t>(pixels.getHeight()),
			static_cast<uint32_t>(pixels.getNumChannels()),
			pixels.getData()
		});
	}
}

inline sd_ctx_params_t buildContextParams(
	const stableDiffusionThread::ContextTaskData& taskData,
	std::vector<std::string>& embeddingNames,
	std::vector<std::string>& embeddingPaths,
	std::vector<sd_embedding_t>& embeddings) {
	sd_ctx_params_t params{};
	sd_ctx_params_init(&params);
	const auto& settings = taskData.contextSettings;

	collectEmbeddings(settings.embedDir, embeddingNames, embeddingPaths, embeddings);

	params.model_path = emptyToNull(settings.modelPath);
	params.clip_l_path = emptyToNull(settings.clipLPath);
	params.clip_g_path = emptyToNull(settings.clipGPath);
	params.t5xxl_path = emptyToNull(settings.t5xxlPath);
	params.diffusion_model_path = emptyToNull(settings.diffusionModelPath);
	params.vae_path = emptyToNull(settings.vaePath);
	params.taesd_path = emptyToNull(settings.taesdPath);
	params.control_net_path = emptyToNull(settings.controlNetPath);
	params.photo_maker_path = emptyToNull(settings.stackedIdEmbedDir);
	params.embeddings = embeddings.empty() ? nullptr : embeddings.data();
	params.embedding_count = static_cast<uint32_t>(embeddings.size());
	params.vae_decode_only = settings.vaeDecodeOnly;
	params.free_params_immediately = settings.freeParamsImmediately;
	params.n_threads = settings.nThreads;
	params.wtype = settings.weightType;
	params.rng_type = settings.rngType;
	params.sampler_rng_type = RNG_TYPE_COUNT;
	params.prediction = settings.prediction;
	params.lora_apply_mode = settings.loraApplyMode;
	params.offload_params_to_cpu = settings.offloadParamsToCpu;
	params.enable_mmap = settings.enableMmap;
	params.keep_clip_on_cpu = settings.keepClipOnCpu;
	params.keep_control_net_on_cpu = settings.keepControlNetCpu;
	params.keep_vae_on_cpu = settings.keepVaeOnCpu;
	params.flash_attn = settings.flashAttn;
	params.diffusion_flash_attn = settings.diffusionFlashAttn;
	return params;
}

inline sd_img_gen_params_t buildImageParams(
	const stableDiffusionThread::ImageTaskData& taskData,
	sd_ctx_t* sdCtx,
	const std::string& effectivePrompt,
	std::vector<sd_lora_t>& loraBuffer,
	std::vector<ofPixels>& pmPixels,
	std::vector<sd_image_t>& pmImageViews) {
	sd_img_gen_params_t params{};
	sd_img_gen_params_init(&params);
	const auto& request = taskData.request;
	const auto& settings = taskData.contextSettings;

	const sample_method_t sampleMethod = resolveSampleMethod(sdCtx, request.sampleMethod);

	params.prompt = emptyToNull(effectivePrompt);
	params.negative_prompt = emptyToNull(request.negativePrompt);
	params.clip_skip = request.clipSkip;
	params.init_image = request.initImage;
	params.mask_image = request.maskImage;
	params.width = request.width;
	params.height = request.height;
	params.sample_params.sample_method = sampleMethod;
	params.sample_params.scheduler = resolveScheduler(sdCtx, sampleMethod, settings.schedule);
	if (request.sampleSteps > 0) {
		params.sample_params.sample_steps = request.sampleSteps;
	}
	if (std::isfinite(request.flowShift)) {
		params.sample_params.flow_shift = request.flowShift;
	}
	if (std::isfinite(request.cfgScale)) {
		params.sample_params.guidance.txt_cfg = request.cfgScale;
	}
	if (std::isfinite(request.cfgScale) &&
		(request.initImage.data != nullptr || !request.instruction.empty())) {
		params.sample_params.guidance.img_cfg = request.cfgScale;
	}
	if (std::isfinite(request.strength)) {
		params.strength = request.strength;
	}
	params.seed = request.seed;
	params.batch_count = request.batchCount;
	params.control_image = request.controlCond ? *request.controlCond : sd_image_t{0, 0, 0, nullptr};
	params.control_strength = request.controlStrength;
	params.vae_tiling_params = makeTilingParams(settings.vaeTiling);

	loraBuffer.clear();
	loraBuffer.reserve(request.loras.size());
	for (const auto& lora : request.loras) {
		if (!lora.isValid()) {
			continue;
		}
		loraBuffer.push_back({lora.isHighNoise, lora.strength, lora.path.c_str()});
	}
	params.loras = loraBuffer.empty() ? nullptr : loraBuffer.data();
	params.lora_count = static_cast<uint32_t>(loraBuffer.size());

	if (!settings.stackedIdEmbedDir.empty() && !request.inputIdImagesPath.empty()) {
		collectPhotoMakerImages(request.inputIdImagesPath, pmPixels, pmImageViews);
		if (!pmImageViews.empty()) {
			params.pm_params.id_images = pmImageViews.data();
			params.pm_params.id_images_count = static_cast<int>(pmImageViews.size());
			params.pm_params.id_embed_path = settings.stackedIdEmbedDir.c_str();
			params.pm_params.style_strength = request.styleStrength;
		}
	}

	return params;
}

inline sd_vid_gen_params_t buildVideoParams(
	const stableDiffusionThread::VideoTaskData& taskData,
	sd_ctx_t* sdCtx,
	std::vector<sd_lora_t>& loraBuffer) {
	sd_vid_gen_params_t params{};
	sd_vid_gen_params_init(&params);
	const auto& request = taskData.request;
	const auto& settings = taskData.contextSettings;

	const sample_method_t sampleMethod = resolveSampleMethod(sdCtx, request.sampleMethod);
	const scheduler_t scheduler = resolveScheduler(sdCtx, sampleMethod, settings.schedule);

	params.prompt = emptyToNull(request.prompt);
	params.negative_prompt = emptyToNull(request.negativePrompt);
	params.clip_skip = request.clipSkip;
	params.init_image = request.initImage;
	params.end_image = request.endImage;
	params.control_frames =
		request.controlFrames.empty() ? nullptr : const_cast<sd_image_t*>(request.controlFrames.data());
	params.control_frames_size = static_cast<int>(request.controlFrames.size());
	params.width = request.width;
	params.height = request.height;
	params.sample_params.sample_method = sampleMethod;
	params.sample_params.scheduler = scheduler;
	if (request.sampleSteps > 0) {
		params.sample_params.sample_steps = request.sampleSteps;
	}
	if (std::isfinite(request.eta)) {
		params.sample_params.eta = request.eta;
	}
	if (std::isfinite(request.flowShift)) {
		params.sample_params.flow_shift = request.flowShift;
	}
	if (std::isfinite(request.cfgScale)) {
		params.sample_params.guidance.txt_cfg = request.cfgScale;
	}
	if (std::isfinite(request.guidance)) {
		params.sample_params.guidance.distilled_guidance = request.guidance;
	}
	if (request.useHighNoiseOverrides) {
		const sample_method_t highNoiseSampleMethod =
			(request.highNoiseSampleMethod == SAMPLE_METHOD_COUNT)
				? sampleMethod
				: resolveSampleMethod(sdCtx, request.highNoiseSampleMethod);
		params.high_noise_sample_params.sample_method = highNoiseSampleMethod;
		params.high_noise_sample_params.scheduler =
			resolveScheduler(sdCtx, highNoiseSampleMethod, settings.schedule);
		if (request.highNoiseSampleSteps > 0) {
			params.high_noise_sample_params.sample_steps = request.highNoiseSampleSteps;
		}
		if (std::isfinite(request.highNoiseEta)) {
			params.high_noise_sample_params.eta = request.highNoiseEta;
		}
		if (std::isfinite(request.highNoiseFlowShift)) {
			params.high_noise_sample_params.flow_shift = request.highNoiseFlowShift;
		}
		if (std::isfinite(request.highNoiseCfgScale)) {
			params.high_noise_sample_params.guidance.txt_cfg = request.highNoiseCfgScale;
		}
		if (std::isfinite(request.highNoiseGuidance)) {
			params.high_noise_sample_params.guidance.distilled_guidance = request.highNoiseGuidance;
		}
	}
	if (std::isfinite(request.moeBoundary)) {
		params.moe_boundary = request.moeBoundary;
	}
	if (std::isfinite(request.strength)) {
		params.strength = request.strength;
	}
	params.seed = request.seed;
	params.video_frames = request.frameCount;
	if (std::isfinite(request.vaceStrength)) {
		params.vace_strength = request.vaceStrength;
	}
	params.vae_tiling_params = makeTilingParams(settings.vaeTiling);
	params.cache = request.cache;

	loraBuffer.clear();
	loraBuffer.reserve(request.loras.size());
	for (const auto& lora : request.loras) {
		if (!lora.isValid()) {
			continue;
		}
		loraBuffer.push_back({lora.isHighNoise, lora.strength, lora.path.c_str()});
	}
	params.loras = loraBuffer.empty() ? nullptr : loraBuffer.data();
	params.lora_count = static_cast<uint32_t>(loraBuffer.size());
	return params;
}

inline std::string formatImageRef(const char* label, const sd_image_t& image) {
	std::ostringstream out;
	out << label << "=";
	if (image.data == nullptr) {
		out << "none";
	} else {
		out << image.width << "x" << image.height << "x" << image.channel;
	}
	return out.str();
}

inline std::string formatScalarOrAuto(float value) {
	if (!std::isfinite(value)) {
		return "auto";
	}
	std::ostringstream out;
	out << value;
	return out.str();
}

inline std::string describeSampleParams(const sd_sample_params_t& params) {
	std::ostringstream out;
	out
		<< "method=" << sd_sample_method_name(params.sample_method)
		<< ", scheduler=" << sd_scheduler_name(params.scheduler)
		<< ", steps=" << params.sample_steps
		<< ", txt_cfg=" << params.guidance.txt_cfg
		<< ", img_cfg=" << formatScalarOrAuto(params.guidance.img_cfg)
		<< ", guidance=" << params.guidance.distilled_guidance
		<< ", slg=" << params.guidance.slg.scale
		<< ", eta=" << formatScalarOrAuto(params.eta)
		<< ", flow_shift=" << formatScalarOrAuto(params.flow_shift);
	return out.str();
}

inline std::string describeHighNoiseSampleParams(const sd_sample_params_t& params) {
	if (params.sample_steps > 0) {
		return describeSampleParams(params);
	}
	if (params.sample_steps < 0) {
		std::ostringstream out;
		out << "enabled(auto";
		if (params.sample_method != SAMPLE_METHOD_COUNT) {
			out << ", method=" << sd_sample_method_name(params.sample_method);
		}
		if (params.scheduler != SCHEDULER_COUNT) {
			out << ", scheduler=" << sd_scheduler_name(params.scheduler);
		}
		out << ")";
		return out.str();
	}
	return "disabled";
}

inline std::string describeVideoParams(const sd_vid_gen_params_t& params) {
	std::ostringstream out;
	out
		<< "prompt=" << (params.prompt ? params.prompt : "")
		<< " | negative=" << (params.negative_prompt ? params.negative_prompt : "")
		<< " | clip_skip=" << params.clip_skip
		<< " | " << formatImageRef("init", params.init_image)
		<< " | " << formatImageRef("end", params.end_image)
		<< " | control_frames=" << params.control_frames_size
		<< " | size=" << params.width << "x" << params.height
		<< " | frames=" << params.video_frames
		<< " | strength=" << params.strength
		<< " | seed=" << params.seed
		<< " | moe_boundary=" << params.moe_boundary
		<< " | vace_strength=" << params.vace_strength
		<< " | sample={" << describeSampleParams(params.sample_params) << "}"
		<< " | high_noise={" << describeHighNoiseSampleParams(params.high_noise_sample_params) << "}"
		<< " | cache_mode=" << static_cast<int>(params.cache.mode)
		<< " | loras=" << params.lora_count;
	return out.str();
}

inline std::string buildResolvedVideoCliCommand(
	const sd_vid_gen_params_t& params,
	const ofxStableDiffusionContextSettings& settings) {
	std::ostringstream command;
	command << "sd-cli.exe -M vid_gen";
	if (!settings.modelPath.empty()) {
		command << " --model " << quoteCliArg(settings.modelPath);
	}
	if (!settings.diffusionModelPath.empty()) {
		command << " --diffusion-model " << quoteCliArg(settings.diffusionModelPath);
	}
	if (!settings.vaePath.empty()) {
		command << " --vae " << quoteCliArg(settings.vaePath);
	}
	if (!settings.t5xxlPath.empty()) {
		command << " --t5xxl " << quoteCliArg(settings.t5xxlPath);
	}
	if (!settings.clipLPath.empty()) {
		command << " --clip_l " << quoteCliArg(settings.clipLPath);
	}
	if (!settings.clipGPath.empty()) {
		command << " --clip_g " << quoteCliArg(settings.clipGPath);
	}
	if (settings.rngType != CUDA_RNG) {
		command << " --rng " << sd_rng_type_name(settings.rngType);
	}
	command << " -p " << quoteCliArg(params.prompt ? params.prompt : "");
	if (params.negative_prompt && params.negative_prompt[0] != '\0') {
		command << " -n " << quoteCliArg(params.negative_prompt);
	}
	if (params.clip_skip != -1) {
		command << " --clip-skip " << params.clip_skip;
	}
	command << " --cfg-scale " << formatCliFloat(params.sample_params.guidance.txt_cfg);
	command << " --guidance " << formatCliFloat(params.sample_params.guidance.distilled_guidance);
	command << " --sampling-method " << sampleMethodToCliName(params.sample_params.sample_method);
	command << " --scheduler " << schedulerToCliName(params.sample_params.scheduler);
	command << " --steps " << params.sample_params.sample_steps;
	command << " -W " << params.width;
	command << " -H " << params.height;
	command << " --video-frames " << params.video_frames;
	if (std::isfinite(params.sample_params.eta)) {
		command << " --eta " << formatCliFloat(params.sample_params.eta);
	}
	if (std::isfinite(params.sample_params.flow_shift)) {
		command << " --flow-shift " << formatCliFloat(params.sample_params.flow_shift);
	}
	if (params.seed >= 0) {
		command << " --seed " << params.seed;
	}
	command << " --moe-boundary " << formatCliFloat(params.moe_boundary);
	if (params.high_noise_sample_params.sample_steps > 0) {
		command << " --high-noise-sampling-method " << sampleMethodToCliName(params.high_noise_sample_params.sample_method);
		command << " --high-noise-steps " << params.high_noise_sample_params.sample_steps;
		command << " --high-noise-cfg-scale " << formatCliFloat(params.high_noise_sample_params.guidance.txt_cfg);
	}
	if (settings.diffusionFlashAttn) {
		command << " --diffusion-fa";
	}
	if (settings.flashAttn) {
		command << " --flash-attn";
	}
	if (settings.offloadParamsToCpu) {
		command << " --offload-to-cpu";
	}
	if (settings.keepVaeOnCpu) {
		command << " --vae-on-cpu";
	}
	if (settings.keepClipOnCpu) {
		command << " --clip-on-cpu";
	}
	if (settings.vaeTiling) {
		command << " --vae-tiling";
	}
	if (params.init_image.data != nullptr) {
		command << " --init-img <loaded input image path not tracked by addon core>";
	}
	if (params.end_image.data != nullptr) {
		command << " --end-img <loaded end frame path not tracked by addon core>";
	}
	if (params.control_frames_size > 0) {
		command << " --control-video <loaded control frame folder not tracked by addon core>";
	}
	if (params.lora_count > 0 && !settings.loraModelDir.empty()) {
		command << " --lora-model-dir " << quoteCliArg(settings.loraModelDir);
	}
	return command.str();
}

}  // namespace ofxStableDiffusionNativeAdapter
