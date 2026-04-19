#include "core/ofxStableDiffusionCapabilityHelpers.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

bool expect(bool condition, const std::string& message) {
	if (condition) {
		return true;
	}

	std::cerr << "FAIL: " << message << std::endl;
	return false;
}

bool expectFamily(
	ofxStableDiffusionModelFamily actual,
	ofxStableDiffusionModelFamily expected,
	const std::string& label) {
	if (actual == expected) {
		return true;
	}

	std::cerr << "FAIL: " << label
		<< " expected " << ofxStableDiffusionModelFamilyLabel(expected)
		<< " but got " << ofxStableDiffusionModelFamilyLabel(actual)
		<< std::endl;
	return false;
}

} // namespace

int main() {
	bool ok = true;

	{
		ofxStableDiffusionContextSettings settings;
		ofxStableDiffusionUpscalerSettings upscaler;
		const auto capabilities =
			ofxStableDiffusionCapabilityHelpers::resolveCapabilities(settings, upscaler);

		ok &= expect(!capabilities.contextConfigured, "empty context is not configured");
		ok &= expect(!capabilities.runtimeResolved, "empty context is not resolved");
		ok &= expectFamily(
			capabilities.modelFamily,
			ofxStableDiffusionModelFamily::Unknown,
			"empty context family");
		ok &= expect(!capabilities.textToImage, "empty context disables text-to-image");
		ok &= expect(!capabilities.imageToVideo, "empty context disables video");
	}

	{
		ofxStableDiffusionContextSettings settings;
		settings.modelPath = "models/sdxl/juggernautXL.safetensors";
		settings.controlNetPath = "models/controlnet/control_v11p_sd15_openpose.safetensors";
		settings.stackedIdEmbedDir = "models/photomaker/photomaker-v2.bin";
		ofxStableDiffusionUpscalerSettings upscaler;

		const auto capabilities =
			ofxStableDiffusionCapabilityHelpers::resolveCapabilities(settings, upscaler);

		ok &= expect(capabilities.contextConfigured, "sdxl context is configured");
		ok &= expect(capabilities.runtimeResolved, "sdxl context resolves runtime support");
		ok &= expectFamily(
			capabilities.modelFamily,
			ofxStableDiffusionModelFamily::SDXL,
			"sdxl family");
		ok &= expect(capabilities.textToImage, "sdxl supports text-to-image");
		ok &= expect(capabilities.imageToImage, "sdxl supports image-to-image");
		ok &= expect(!capabilities.imageToVideo, "sdxl does not expose video generation");
		ok &= expect(capabilities.controlNetConfigured, "sdxl controlnet path is tracked");
		ok &= expect(capabilities.controlNet, "sdxl resolves controlnet support");
		ok &= expect(capabilities.photoMakerConfigured, "sdxl photomaker path is tracked");
		ok &= expect(capabilities.photoMaker, "sdxl resolves photomaker support");
		ok &= expect(!capabilities.videoEndFrame, "sdxl does not expose end-frame morphing");
	}

	{
		ofxStableDiffusionContextSettings settings;
		settings.modelPath = "models/video/wan2.1-flf2v-14b-Q4_K_M.gguf";
		ofxStableDiffusionUpscalerSettings upscaler;

		const auto capabilities =
			ofxStableDiffusionCapabilityHelpers::resolveCapabilities(settings, upscaler);

		ok &= expectFamily(
			capabilities.modelFamily,
			ofxStableDiffusionModelFamily::WANFLF2V,
			"wan flf2v family");
		ok &= expect(!capabilities.textToImage, "wan flf2v disables image generation modes");
		ok &= expect(capabilities.imageToVideo, "wan flf2v supports image-to-video");
		ok &= expect(capabilities.videoEndFrame, "wan flf2v supports end-frame morphing");
		ok &= expect(capabilities.videoAnimation, "wan flf2v supports wrapper animation path");
		ok &= expect(!capabilities.controlNet, "wan flf2v does not resolve controlnet support");
		ok &= expect(!capabilities.photoMaker, "wan flf2v does not resolve photomaker support");
	}

	{
		ofxStableDiffusionContextSettings settings;
		settings.modelPath = "models/video/Wan2.2-TI2V-5B.gguf";
		ofxStableDiffusionUpscalerSettings upscaler;

		const auto capabilities =
			ofxStableDiffusionCapabilityHelpers::resolveCapabilities(settings, upscaler);

		ok &= expectFamily(
			capabilities.modelFamily,
			ofxStableDiffusionModelFamily::WANTI2V,
			"wan ti2v family");
		ok &= expect(capabilities.imageToVideo, "wan ti2v supports video generation");
		ok &= expect(!capabilities.videoEndFrame, "wan ti2v does not advertise end-frame morphing");
	}

	{
		ofxStableDiffusionContextSettings settings;
		settings.modelPath = "models/flux/flux-controls-dev.safetensors";
		ofxStableDiffusionUpscalerSettings upscaler;

		const auto capabilities =
			ofxStableDiffusionCapabilityHelpers::resolveCapabilities(settings, upscaler);

		ok &= expectFamily(
			capabilities.modelFamily,
			ofxStableDiffusionModelFamily::FLUXControl,
			"flux control family");
		ok &= expect(capabilities.nativeControlModel, "flux control family marks native control support");
		ok &= expect(capabilities.controlNet, "flux control family resolves control support without external path");
		ok &= expect(capabilities.textToImage, "flux control still supports image generation");
	}

	{
		ofxStableDiffusionContextSettings settings;
		settings.diffusionModelPath = "models/flux/flux1-dev-Q8_0.gguf";
		settings.clipLPath = "models/flux/clip_l.safetensors";
		settings.t5xxlPath = "models/flux/t5xxl_fp16.safetensors";
		ofxStableDiffusionUpscalerSettings upscaler;
		upscaler.enabled = true;
		upscaler.modelPath = "models/esrgan/4x.pth";

		const auto capabilities =
			ofxStableDiffusionCapabilityHelpers::resolveCapabilities(settings, upscaler);

		ok &= expectFamily(
			capabilities.modelFamily,
			ofxStableDiffusionModelFamily::FLUX,
			"split flux family");
		ok &= expect(capabilities.splitModelPaths, "split flux settings are detected");
		ok &= expect(capabilities.textToImage, "split flux supports image generation");
		ok &= expect(!capabilities.imageToVideo, "split flux does not advertise video generation");
		ok &= expect(capabilities.upscaling, "configured upscaler is reflected in capabilities");
	}

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
