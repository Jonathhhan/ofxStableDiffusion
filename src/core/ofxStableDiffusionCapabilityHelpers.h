#pragma once

#include "ofxStableDiffusionStringUtils.h"
#include "ofxStableDiffusionTypes.h"

#include <initializer_list>
#include <string>

inline const char * ofxStableDiffusionModelFamilyLabel(ofxStableDiffusionModelFamily family) {
	switch (family) {
	case ofxStableDiffusionModelFamily::SD1: return "SD1";
	case ofxStableDiffusionModelFamily::SD2: return "SD2";
	case ofxStableDiffusionModelFamily::SDXL: return "SDXL";
	case ofxStableDiffusionModelFamily::SD3: return "SD3";
	case ofxStableDiffusionModelFamily::FLUX: return "FLUX";
	case ofxStableDiffusionModelFamily::FLUXFill: return "FLUXFill";
	case ofxStableDiffusionModelFamily::FLUXControl: return "FLUXControl";
	case ofxStableDiffusionModelFamily::FLUX2: return "FLUX2";
	case ofxStableDiffusionModelFamily::WAN: return "WAN";
	case ofxStableDiffusionModelFamily::WANI2V: return "WANI2V";
	case ofxStableDiffusionModelFamily::WANTI2V: return "WANTI2V";
	case ofxStableDiffusionModelFamily::WANFLF2V: return "WANFLF2V";
	case ofxStableDiffusionModelFamily::WANVACE: return "WANVACE";
	case ofxStableDiffusionModelFamily::Unknown:
	default:
		return "Unknown";
	}
}

namespace ofxStableDiffusionCapabilityHelpers {

inline bool containsAny(
	const std::string& haystack,
	std::initializer_list<const char*> needles) {
	for (const char* needle : needles) {
		if (haystack.find(needle) != std::string::npos) {
			return true;
		}
	}
	return false;
}

inline bool hasConfiguredModelPaths(const ofxStableDiffusionContextSettings& settings) {
	return !settings.modelPath.empty() ||
		!settings.diffusionModelPath.empty() ||
		!settings.clipLPath.empty() ||
		!settings.clipGPath.empty() ||
		!settings.t5xxlPath.empty();
}

inline std::string buildPrimaryModelDescriptor(const ofxStableDiffusionContextSettings& settings) {
	if (!settings.modelPath.empty()) {
		return ofxSdToLowerCopy(settings.modelPath);
	}
	if (!settings.diffusionModelPath.empty()) {
		return ofxSdToLowerCopy(settings.diffusionModelPath);
	}
	if (!settings.clipLPath.empty()) {
		return ofxSdToLowerCopy(settings.clipLPath);
	}
	if (!settings.clipGPath.empty()) {
		return ofxSdToLowerCopy(settings.clipGPath);
	}
	return ofxSdToLowerCopy(settings.t5xxlPath);
}

inline std::string buildAllModelDescriptors(const ofxStableDiffusionContextSettings& settings) {
	return ofxSdToLowerCopy(
		settings.modelPath + " " +
		settings.diffusionModelPath + " " +
		settings.clipLPath + " " +
		settings.clipGPath + " " +
		settings.t5xxlPath);
}

inline ofxStableDiffusionModelFamily inferModelFamily(const ofxStableDiffusionContextSettings& settings) {
	const std::string primary = buildPrimaryModelDescriptor(settings);
	const std::string all = buildAllModelDescriptors(settings);
	const auto hasHint = [&primary, &all](std::initializer_list<const char*> hints) {
		return containsAny(primary, hints) || containsAny(all, hints);
	};

	if (hasHint({"wan", "wanvideo"})) {
		if (hasHint({"vace"})) {
			return ofxStableDiffusionModelFamily::WANVACE;
		}
		if (hasHint({"flf2v"})) {
			return ofxStableDiffusionModelFamily::WANFLF2V;
		}
		if (hasHint({"ti2v"})) {
			return ofxStableDiffusionModelFamily::WANTI2V;
		}
		if (hasHint({"i2v"})) {
			return ofxStableDiffusionModelFamily::WANI2V;
		}
		return ofxStableDiffusionModelFamily::WAN;
	}

	if (hasHint({"flux2", "flux-2", "flux_2", "klein"})) {
		return ofxStableDiffusionModelFamily::FLUX2;
	}

	if (hasHint({"flux", "flex.2", "flex-2", "flex_2"})) {
		if (hasHint({"control", "controls", "flex.2", "flex-2", "flex_2"})) {
			return ofxStableDiffusionModelFamily::FLUXControl;
		}
		if (hasHint({"fill", "inpaint"})) {
			return ofxStableDiffusionModelFamily::FLUXFill;
		}
		return ofxStableDiffusionModelFamily::FLUX;
	}

	if (hasHint({"sd3", "stable-diffusion-3"})) {
		return ofxStableDiffusionModelFamily::SD3;
	}

	if (hasHint({"sdxl", "xl-base", "juggernautxl", "ponyxl", "xl\\", "xl/"})) {
		return ofxStableDiffusionModelFamily::SDXL;
	}

	if (hasHint({"sd2", "v2-1", "v2.1", "2.1"})) {
		return ofxStableDiffusionModelFamily::SD2;
	}

	if (hasHint({"sd1", "sd15", "v1-5", "v1.5", "1.5"})) {
		return ofxStableDiffusionModelFamily::SD1;
	}

	if (!settings.diffusionModelPath.empty() &&
		(!settings.clipLPath.empty() || !settings.t5xxlPath.empty())) {
		return ofxStableDiffusionModelFamily::FLUX;
	}

	return ofxStableDiffusionModelFamily::Unknown;
}

inline bool familySupportsImageGeneration(ofxStableDiffusionModelFamily family) {
	switch (family) {
	case ofxStableDiffusionModelFamily::WANI2V:
	case ofxStableDiffusionModelFamily::WANTI2V:
	case ofxStableDiffusionModelFamily::WANFLF2V:
	case ofxStableDiffusionModelFamily::WANVACE:
	case ofxStableDiffusionModelFamily::WAN:
		return false;
	case ofxStableDiffusionModelFamily::Unknown:
	default:
		return true;
	}
}

inline bool familySupportsVideo(ofxStableDiffusionModelFamily family) {
	switch (family) {
	case ofxStableDiffusionModelFamily::WAN:
	case ofxStableDiffusionModelFamily::WANI2V:
	case ofxStableDiffusionModelFamily::WANTI2V:
	case ofxStableDiffusionModelFamily::WANFLF2V:
	case ofxStableDiffusionModelFamily::WANVACE:
		return true;
	default:
		return false;
	}
}

inline bool familyRequiresInputImageForVideo(ofxStableDiffusionModelFamily family) {
	switch (family) {
	case ofxStableDiffusionModelFamily::WANI2V:
	case ofxStableDiffusionModelFamily::WANTI2V:
	case ofxStableDiffusionModelFamily::WANFLF2V:
	case ofxStableDiffusionModelFamily::WANVACE:
		return true;
	default:
		return false;
	}
}

inline bool familySupportsVideoEndFrame(ofxStableDiffusionModelFamily family) {
	switch (family) {
	case ofxStableDiffusionModelFamily::WANI2V:
	case ofxStableDiffusionModelFamily::WANFLF2V:
		return true;
	default:
		return false;
	}
}

inline bool familySupportsWrapperVideoAnimation(ofxStableDiffusionModelFamily family) {
	switch (family) {
	case ofxStableDiffusionModelFamily::WANFLF2V:
		return true;
	case ofxStableDiffusionModelFamily::WAN:
	case ofxStableDiffusionModelFamily::WANI2V:
	case ofxStableDiffusionModelFamily::WANTI2V:
	case ofxStableDiffusionModelFamily::WANVACE:
		return false;
	default:
		return false;
	}
}

inline bool familySupportsPhotoMaker(ofxStableDiffusionModelFamily family) {
	return family == ofxStableDiffusionModelFamily::SDXL;
}

inline bool familyHasNativeControlModel(ofxStableDiffusionModelFamily family) {
	return family == ofxStableDiffusionModelFamily::FLUXControl;
}

inline bool familySupportsExternalControlNet(ofxStableDiffusionModelFamily family) {
	switch (family) {
	case ofxStableDiffusionModelFamily::WANI2V:
	case ofxStableDiffusionModelFamily::WANTI2V:
	case ofxStableDiffusionModelFamily::WANFLF2V:
	case ofxStableDiffusionModelFamily::WANVACE:
	case ofxStableDiffusionModelFamily::WAN:
		return false;
	default:
		return true;
	}
}

inline ofxStableDiffusionCapabilities resolveCapabilities(
	const ofxStableDiffusionContextSettings& settings,
	const ofxStableDiffusionUpscalerSettings& upscalerSettings) {
	ofxStableDiffusionCapabilities capabilities;
	capabilities.contextConfigured = hasConfiguredModelPaths(settings);
	capabilities.runtimeResolved = capabilities.contextConfigured;
	capabilities.modelFamily = inferModelFamily(settings);
	capabilities.splitModelPaths =
		!settings.diffusionModelPath.empty() ||
		!settings.clipLPath.empty() ||
		!settings.clipGPath.empty() ||
		!settings.t5xxlPath.empty();
	capabilities.controlNetConfigured = !settings.controlNetPath.empty();
	capabilities.photoMakerConfigured = !settings.stackedIdEmbedDir.empty();
	capabilities.nativeControlModel = familyHasNativeControlModel(capabilities.modelFamily);
	capabilities.upscaling = upscalerSettings.enabled && !upscalerSettings.modelPath.empty();
	capabilities.mmap = settings.enableMmap;
	capabilities.flashAttention = settings.flashAttn;
	capabilities.videoMetadataExport = true;

	if (!capabilities.contextConfigured) {
		return capabilities;
	}

	const bool imageGeneration = familySupportsImageGeneration(capabilities.modelFamily);
	capabilities.textToImage = imageGeneration;
	capabilities.imageToImage = imageGeneration;
	capabilities.inpainting = imageGeneration;
	capabilities.imageToVideo = familySupportsVideo(capabilities.modelFamily);
	capabilities.videoRequiresInputImage =
		familyRequiresInputImageForVideo(capabilities.modelFamily);
	capabilities.videoEndFrame = familySupportsVideoEndFrame(capabilities.modelFamily);
	capabilities.videoAnimation = familySupportsWrapperVideoAnimation(capabilities.modelFamily);
	capabilities.lora = imageGeneration || capabilities.imageToVideo;
	capabilities.embeddings = imageGeneration;
	capabilities.controlNet =
		capabilities.nativeControlModel ||
		(capabilities.controlNetConfigured &&
			familySupportsExternalControlNet(capabilities.modelFamily));
	capabilities.photoMaker =
		capabilities.photoMakerConfigured &&
		familySupportsPhotoMaker(capabilities.modelFamily);

	return capabilities;
}

} // namespace ofxStableDiffusionCapabilityHelpers
