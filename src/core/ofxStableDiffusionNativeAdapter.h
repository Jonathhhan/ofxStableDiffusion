#pragma once

#include "../ofxStableDiffusionThread.h"

#include <string>
#include <utility>
#include <vector>

namespace ofxStableDiffusionNativeAdapter {

inline const char* emptyToNull(const std::string& value) {
	return value.empty() ? nullptr : value.c_str();
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
	params.sampler_rng_type = settings.rngType;
	params.prediction = settings.prediction;
	params.lora_apply_mode = settings.loraApplyMode;
	params.offload_params_to_cpu = settings.offloadParamsToCpu;
	params.enable_mmap = settings.enableMmap;
	params.keep_clip_on_cpu = settings.keepClipOnCpu;
	params.keep_control_net_on_cpu = settings.keepControlNetCpu;
	params.keep_vae_on_cpu = settings.keepVaeOnCpu;
	params.flash_attn = settings.flashAttn;
	params.diffusion_flash_attn = settings.flashAttn;
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
	params.sample_params.sample_steps = request.sampleSteps;
	params.sample_params.guidance.txt_cfg = request.cfgScale;
	if (request.initImage.data != nullptr || !request.instruction.empty()) {
		params.sample_params.guidance.img_cfg = request.cfgScale;
	}
	params.strength = request.strength;
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
	params.width = request.width;
	params.height = request.height;
	params.sample_params.sample_method = sampleMethod;
	params.sample_params.scheduler = scheduler;
	params.sample_params.sample_steps = request.sampleSteps;
	params.sample_params.guidance.txt_cfg = request.cfgScale;
	params.high_noise_sample_params = params.sample_params;
	params.strength = request.strength;
	params.seed = request.seed;
	params.video_frames = request.frameCount;
	params.vace_strength = request.vaceStrength;
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
	return params;
}

}  // namespace ofxStableDiffusionNativeAdapter
