#include "ofxStableDiffusionThread.h"
#include "ofxStableDiffusion.h"
#include "core/ofxStableDiffusionMemoryHelpers.h"

#include <vector>

namespace {

const char* emptyToNull(const std::string& value) {
	return value.empty() ? nullptr : value.c_str();
}

sample_method_t resolveSampleMethod(sd_ctx_t* sdCtx, sample_method_t requested) {
	if (requested == SAMPLE_METHOD_COUNT) {
		return sd_get_default_sample_method(sdCtx);
	}
	return requested;
}

scheduler_t resolveScheduler(
	sd_ctx_t* sdCtx,
	sample_method_t sampleMethod,
	scheduler_t requestedSchedule) {
	if (requestedSchedule == SCHEDULER_COUNT) {
		return sd_get_default_scheduler(sdCtx, sampleMethod);
	}
	return requestedSchedule;
}

sd_tiling_params_t makeTilingParams(bool enabled) {
	sd_tiling_params_t tiling{};
	tiling.enabled = enabled;
	tiling.target_overlap = 0.5f;
	return tiling;
}

std::string buildEffectivePrompt(const ofxStableDiffusion* sd) {
	if (sd->instruction.empty()) {
		return sd->prompt;
	}
	if (sd->prompt.empty()) {
		return sd->instruction;
	}
	if (sd->instruction == sd->prompt) {
		return sd->prompt;
	}
	return sd->prompt + "\nInstruction: " + sd->instruction;
}

bool looksLikeEmbeddingFile(const std::string& extension) {
	return extension == "pt" ||
		extension == "ckpt" ||
		extension == "safetensors" ||
		extension == "bin" ||
		extension == "gguf";
}

void collectEmbeddings(
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

bool looksLikeImageFile(const std::string& extension) {
	return extension == "png" ||
		extension == "jpg" ||
		extension == "jpeg" ||
		extension == "bmp" ||
		extension == "webp";
}

void collectPhotoMakerImages(
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

sd_ctx_params_t buildContextParams(
	ofxStableDiffusion* sd,
	std::vector<std::string>& embeddingNames,
	std::vector<std::string>& embeddingPaths,
	std::vector<sd_embedding_t>& embeddings) {
	sd_ctx_params_t params{};
	sd_ctx_params_init(&params);

	collectEmbeddings(sd->embedDirCStr, embeddingNames, embeddingPaths, embeddings);

	params.model_path = emptyToNull(sd->modelPath);
	params.clip_l_path = emptyToNull(sd->clipLPath);
	params.clip_g_path = emptyToNull(sd->clipGPath);
	params.t5xxl_path = emptyToNull(sd->t5xxlPath);
	params.diffusion_model_path = emptyToNull(sd->diffusionModelPath);
	params.vae_path = emptyToNull(sd->vaePath);
	params.taesd_path = emptyToNull(sd->taesdPath);
	params.control_net_path = emptyToNull(sd->controlNetPathCStr);
	params.photo_maker_path = emptyToNull(sd->stackedIdEmbedDirCStr);
	params.embeddings = embeddings.empty() ? nullptr : embeddings.data();
	params.embedding_count = static_cast<uint32_t>(embeddings.size());
	params.vae_decode_only = sd->vaeDecodeOnly;
	params.free_params_immediately = sd->freeParamsImmediately;
	params.n_threads = sd->nThreads;
	params.wtype = sd->wType;
	params.rng_type = sd->rngType;
	params.sampler_rng_type = sd->rngType;
	params.prediction = sd->prediction;
	params.lora_apply_mode = sd->loraApplyMode;
	params.offload_params_to_cpu = sd->offloadParamsToCpu;
	params.enable_mmap = sd->enableMmap;
	params.keep_clip_on_cpu = sd->keepClipOnCpu;
	params.keep_control_net_on_cpu = sd->keepControlNetCpu;
	params.keep_vae_on_cpu = sd->keepVaeOnCpu;
	params.flash_attn = sd->flashAttn;
	params.diffusion_flash_attn = sd->flashAttn;
	return params;
}

sd_img_gen_params_t buildImageParams(
	ofxStableDiffusion* sd,
	sd_ctx_t* sdCtx,
	const std::string& effectivePrompt,
	std::vector<sd_lora_t>& loraBuffer,
	std::vector<ofPixels>& pmPixels,
	std::vector<sd_image_t>& pmImageViews) {
	sd_img_gen_params_t params{};
	sd_img_gen_params_init(&params);

	const sample_method_t sampleMethod = resolveSampleMethod(sdCtx, sd->sampleMethodEnum);

	params.prompt = emptyToNull(effectivePrompt);
	params.negative_prompt = emptyToNull(sd->negativePrompt);
	params.clip_skip = sd->clipSkip;
	params.init_image = sd->inputImage;
	params.mask_image = sd->maskImage;
	params.width = sd->width;
	params.height = sd->height;
	params.sample_params.sample_method = sampleMethod;
	params.sample_params.scheduler = resolveScheduler(sdCtx, sampleMethod, sd->schedule);
	params.sample_params.sample_steps = sd->sampleSteps;
	params.sample_params.guidance.txt_cfg = sd->cfgScale;
	if (sd->inputImage.data != nullptr || !sd->instruction.empty()) {
		params.sample_params.guidance.img_cfg = sd->cfgScale;
	}
	params.strength = sd->strength;
	params.seed = sd->seed;
	params.batch_count = sd->batchCount;
	params.control_image = sd->controlCond ? *sd->controlCond : sd_image_t{0, 0, 0, nullptr};
	params.control_strength = sd->controlStrength;
	params.vae_tiling_params = makeTilingParams(sd->vaeTiling);

	loraBuffer.clear();
	loraBuffer.reserve(sd->loras.size());
	for (const auto& lora : sd->loras) {
		if (!lora.isValid()) {
			continue;
		}
		loraBuffer.push_back({lora.isHighNoise, lora.strength, lora.path.c_str()});
	}
	params.loras = loraBuffer.empty() ? nullptr : loraBuffer.data();
	params.lora_count = static_cast<uint32_t>(loraBuffer.size());

	if (!sd->stackedIdEmbedDirCStr.empty() && !sd->inputIdImagesPath.empty()) {
		collectPhotoMakerImages(sd->inputIdImagesPath, pmPixels, pmImageViews);
		if (!pmImageViews.empty()) {
			params.pm_params.id_images = pmImageViews.data();
			params.pm_params.id_images_count = static_cast<int>(pmImageViews.size());
			params.pm_params.id_embed_path = sd->stackedIdEmbedDirCStr.c_str();
			params.pm_params.style_strength = sd->styleStrength;
		}
	}

	return params;
}

sd_vid_gen_params_t buildVideoParams(ofxStableDiffusion* sd, sd_ctx_t* sdCtx, std::vector<sd_lora_t>& loraBuffer) {
	sd_vid_gen_params_t params{};
	sd_vid_gen_params_init(&params);

	const sample_method_t sampleMethod = resolveSampleMethod(sdCtx, sd->sampleMethodEnum);
	const scheduler_t scheduler = resolveScheduler(sdCtx, sampleMethod, sd->schedule);

	params.prompt = emptyToNull(sd->prompt);
	params.negative_prompt = emptyToNull(sd->negativePrompt);
	params.clip_skip = sd->clipSkip;
	params.init_image = sd->inputImage;
	params.end_image = sd->endImage;
	params.width = sd->width;
	params.height = sd->height;
	params.sample_params.sample_method = sampleMethod;
	params.sample_params.scheduler = scheduler;
	params.sample_params.sample_steps = sd->sampleSteps;
	params.sample_params.guidance.txt_cfg = sd->cfgScale;
	params.high_noise_sample_params = params.sample_params;
	params.strength = sd->strength;
	params.seed = sd->seed;
	params.video_frames = sd->videoFrames;
	params.vace_strength = sd->vaceStrength;
	params.vae_tiling_params = makeTilingParams(sd->vaeTiling);
	loraBuffer.clear();
	loraBuffer.reserve(sd->loras.size());
	for (const auto& lora : sd->loras) {
		if (!lora.isValid()) {
			continue;
		}
		loraBuffer.push_back({lora.isHighNoise, lora.strength, lora.path.c_str()});
	}
	params.loras = loraBuffer.empty() ? nullptr : loraBuffer.data();
	params.lora_count = static_cast<uint32_t>(loraBuffer.size());
	return params;
}

} // namespace

stableDiffusionThread::~stableDiffusionThread() {
	if (isThreadRunning()) {
		waitForThread(true);
	}
	clearContexts();
}

void stableDiffusionThread::clearContexts() {
	if (sdCtx) {
		free_sd_ctx(sdCtx);
		sdCtx = nullptr;
	}
	if (upscalerCtx) {
		free_upscaler_ctx(upscalerCtx);
		upscalerCtx = nullptr;
	}
	isSdCtxLoaded = false;
	isUpscalerCtxLoaded = false;
}

void stableDiffusionThread::threadedFunction() {
	ofxStableDiffusion* sd = static_cast<ofxStableDiffusion*>(userData);
	if (!sd) {
		return;
	}

	if (sd->activeTask == ofxStableDiffusionTask::LoadModel || sd->isModelLoading) {
		if (sdCtx) {
			free_sd_ctx(sdCtx);
			sdCtx = nullptr;
		}

		std::vector<std::string> embeddingNames;
		std::vector<std::string> embeddingPaths;
		std::vector<sd_embedding_t> embeddings;
		sd_ctx_params_t ctxParams =
			buildContextParams(sd, embeddingNames, embeddingPaths, embeddings);
		sdCtx = new_sd_ctx(&ctxParams);

		if (upscalerCtx) {
			free_upscaler_ctx(upscalerCtx);
			upscalerCtx = nullptr;
			isUpscalerCtxLoaded = false;
		}
		if (sd->isESRGAN && !sd->esrganPath.empty()) {
			upscalerCtx = new_upscaler_ctx(sd->esrganPath.c_str(), false, false, sd->nThreads, 0);
			isUpscalerCtxLoaded = (upscalerCtx != nullptr);
		}
		isSdCtxLoaded = (sdCtx != nullptr);
		sd->isModelLoading = false;
		sd->activeTask = ofxStableDiffusionTask::None;
		if (!isSdCtxLoaded) {
			sd->setLastError("Failed to create stable-diffusion context");
		}
		return;
	}

	if (!sdCtx) {
		sd->activeTask = ofxStableDiffusionTask::None;
		sd->setLastError("Stable Diffusion context is not loaded");
		return;
	}

	if (sd->activeTask == ofxStableDiffusionTask::ImageToVideo || sd->isImageToVideo) {
		sd_vid_gen_params_t params = buildVideoParams(sd, sdCtx, loraBuffer);
		int generatedFrameCount = 0;
		sd_image_t* output = generate_video(sdCtx, &params, &generatedFrameCount);
		const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
		if (!output || generatedFrameCount <= 0) {
			ofxSdReleaseImageArray(output, generatedFrameCount);
			sd->setLastError("Image-to-video generation returned no frames");
			sd->activeTask = ofxStableDiffusionTask::None;
			return;
		}
		sd->captureVideoResults(output, generatedFrameCount, params.seed, elapsedMs);
		sd->activeTask = ofxStableDiffusionTask::None;
		return;
	}

	const std::string effectivePrompt = buildEffectivePrompt(sd);
	std::vector<ofPixels> pmPixels;
	std::vector<sd_image_t> pmImageViews;
	sd_img_gen_params_t params =
		buildImageParams(sd, sdCtx, effectivePrompt, loraBuffer, pmPixels, pmImageViews);
	sd_image_t* output = generate_image(sdCtx, &params);

	if (output && sd->isESRGAN) {
		if (!upscalerCtx) {
			ofxSdReleaseImageArray(output, sd->batchCount);
			sd->setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Upscaler context is not loaded");
			sd->activeTask = ofxStableDiffusionTask::None;
			return;
		}

		for (int i = 0; i < sd->batchCount; i++) {
			sd_image_t upscaled = upscale(upscalerCtx, output[i], sd->esrganMultiplier);
			if (!upscaled.data) {
				ofxSdReleaseImage(output[i]);
				ofxSdReleaseImageArray(output, sd->batchCount);
				sd->setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Upscaling failed for one or more images");
				sd->activeTask = ofxStableDiffusionTask::None;
				return;
			}
			ofxSdReleaseImage(output[i]);
			output[i] = upscaled;
		}
	}

	const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
	if (!output) {
		sd->setLastError("Image generation returned no images");
		sd->activeTask = ofxStableDiffusionTask::None;
		return;
	}

	sd->captureImageResults(output, sd->batchCount, params.seed, elapsedMs);
	sd->activeTask = ofxStableDiffusionTask::None;
}
