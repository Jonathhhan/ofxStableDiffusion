#pragma once

#include "ofxStableDiffusionEnums.h"

inline const char * ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode mode) {
	switch (mode) {
	case ofxStableDiffusionImageMode::ImageToImage: return "ImageToImage";
	case ofxStableDiffusionImageMode::InstructImage: return "InstructImage";
	case ofxStableDiffusionImageMode::Variation: return "Variation";
	case ofxStableDiffusionImageMode::Restyle: return "Restyle";
	case ofxStableDiffusionImageMode::Inpainting: return "Inpainting";
	case ofxStableDiffusionImageMode::TextToImage:
	default:
		return "TextToImage";
	}
}

inline bool ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode mode) {
	return mode != ofxStableDiffusionImageMode::TextToImage;
}

inline ofxStableDiffusionTask ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode mode) {
	switch (mode) {
	case ofxStableDiffusionImageMode::InstructImage:
		return ofxStableDiffusionTask::InstructImage;
	case ofxStableDiffusionImageMode::Variation:
		return ofxStableDiffusionTask::ImageVariation;
	case ofxStableDiffusionImageMode::Restyle:
		return ofxStableDiffusionTask::ImageRestyle;
	case ofxStableDiffusionImageMode::Inpainting:
		return ofxStableDiffusionTask::Inpainting;
	case ofxStableDiffusionImageMode::ImageToImage:
		return ofxStableDiffusionTask::ImageToImage;
	case ofxStableDiffusionImageMode::TextToImage:
	default:
		return ofxStableDiffusionTask::TextToImage;
	}
}

inline float ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode mode) {
	switch (mode) {
	case ofxStableDiffusionImageMode::InstructImage: return 0.35f;
	case ofxStableDiffusionImageMode::Variation: return 0.25f;
	case ofxStableDiffusionImageMode::Restyle: return 0.75f;
	case ofxStableDiffusionImageMode::Inpainting: return 0.75f;
	case ofxStableDiffusionImageMode::ImageToImage: return 0.50f;
	case ofxStableDiffusionImageMode::TextToImage:
	default:
		return 0.50f;
	}
}

inline float ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode mode) {
	switch (mode) {
	case ofxStableDiffusionImageMode::InstructImage: return 4.5f;
	case ofxStableDiffusionImageMode::Variation: return 3.0f;
	case ofxStableDiffusionImageMode::Restyle: return 9.0f;
	case ofxStableDiffusionImageMode::Inpainting: return 7.5f;
	case ofxStableDiffusionImageMode::ImageToImage: return 7.0f;
	case ofxStableDiffusionImageMode::TextToImage:
	default:
		return 7.0f;
	}
}
