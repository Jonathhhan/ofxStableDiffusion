#pragma once

#include "ofxStableDiffusionLongVideoManifest.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>

class ofxStableDiffusion;

namespace ofxStableDiffusionLongVideoWorkflow {

inline const char* presetLabel(ofxStableDiffusionLongVideoPreset preset) {
	switch (preset) {
	case ofxStableDiffusionLongVideoPreset::FastPreview: return "FastPreview";
	case ofxStableDiffusionLongVideoPreset::LowVram: return "LowVram";
	case ofxStableDiffusionLongVideoPreset::Balanced: return "Balanced";
	case ofxStableDiffusionLongVideoPreset::Quality: return "Quality";
	case ofxStableDiffusionLongVideoPreset::BatchStoryboard: return "BatchStoryboard";
	default:
		return "Balanced";
	}
}

inline void applyPreset(
	ofxStableDiffusionVideoRequest* request,
	ofxStableDiffusionLongVideoPreset preset) {
	if (!request) {
		return;
	}

	switch (preset) {
	case ofxStableDiffusionLongVideoPreset::FastPreview:
		request->frameCount = 8;
		request->fps = 8;
		request->sampleSteps = 12;
		request->cfgScale = 5.0f;
		request->strength = 0.55f;
		break;
	case ofxStableDiffusionLongVideoPreset::LowVram:
		request->frameCount = 10;
		request->fps = 8;
		request->sampleSteps = 14;
		request->cfgScale = 5.5f;
		request->strength = 0.6f;
		request->width = std::min(request->width, 512);
		request->height = std::min(request->height, 768);
		break;
	case ofxStableDiffusionLongVideoPreset::Balanced:
		request->frameCount = 12;
		request->fps = 10;
		request->sampleSteps = 20;
		request->cfgScale = 6.5f;
		request->strength = 0.65f;
		break;
	case ofxStableDiffusionLongVideoPreset::Quality:
		request->frameCount = 16;
		request->fps = 12;
		request->sampleSteps = 28;
		request->cfgScale = 7.0f;
		request->strength = 0.7f;
		break;
	case ofxStableDiffusionLongVideoPreset::BatchStoryboard:
		request->frameCount = 6;
		request->fps = 6;
		request->sampleSteps = 16;
		request->cfgScale = 6.0f;
		request->strength = 0.6f;
		break;
	}
}

inline std::string trimCopy(const std::string& value) {
	std::size_t start = 0;
	while (start < value.size() &&
		std::isspace(static_cast<unsigned char>(value[start]))) {
		++start;
	}
	std::size_t end = value.size();
	while (end > start &&
		std::isspace(static_cast<unsigned char>(value[end - 1]))) {
		--end;
	}
	return value.substr(start, end - start);
}

inline std::string joinPath(const std::string& left, const std::string& right) {
	if (left.empty()) {
		return right;
	}
	const char last = left.back();
	if (last == '/' || last == '\\') {
		return left + right;
	}
	return left + "/" + right;
}

inline std::string jsonEscape(const std::string& value) {
	std::ostringstream output;
	for (const char ch : value) {
		switch (ch) {
		case '\\': output << "\\\\"; break;
		case '"': output << "\\\""; break;
		case '\n': output << "\\n"; break;
		case '\r': output << "\\r"; break;
		case '\t': output << "\\t"; break;
		default: output << ch; break;
		}
	}
	return output.str();
}

inline ofxStableDiffusionLongVideoValidation validate(
	const ofxStableDiffusionLongVideoManifest& manifest) {
	ofxStableDiffusionLongVideoValidation validation;
	if (trimCopy(manifest.projectName).empty()) {
		validation.ok = false;
		validation.errors.push_back("Project name is empty.");
	}
	if (manifest.chunks.empty()) {
		validation.ok = false;
		validation.errors.push_back("Long-video manifest has no chunks.");
	}
	if (trimCopy(manifest.outputDirectory).empty()) {
		validation.warnings.push_back("Output directory is empty; render orchestration should set one before export.");
	}
	for (std::size_t i = 0; i < manifest.chunks.size(); ++i) {
		const auto& chunk = manifest.chunks[i];
		if (trimCopy(chunk.prompt).empty()) {
			validation.ok = false;
			validation.errors.push_back("Chunk " + std::to_string(i + 1) + " has an empty prompt.");
		}
		if (chunk.width <= 0 || chunk.height <= 0) {
			validation.ok = false;
			validation.errors.push_back("Chunk " + std::to_string(i + 1) + " has invalid dimensions.");
		}
		if (chunk.frameCount <= 0 || chunk.fps <= 0) {
			validation.ok = false;
			validation.errors.push_back("Chunk " + std::to_string(i + 1) + " has invalid frame count or FPS.");
		}
		if (chunk.sampleSteps <= 0) {
			validation.ok = false;
			validation.errors.push_back("Chunk " + std::to_string(i + 1) + " has invalid sample steps.");
		}
		if (chunk.frameCount < 8) {
			validation.warnings.push_back("Chunk " + std::to_string(i + 1) + " is very short and may feel abrupt.");
		}
	}
	return validation;
}

inline ofxStableDiffusionVideoRequest buildChunkRequest(
	const ofxStableDiffusionLongVideoManifest& manifest,
	const ofxStableDiffusionLongVideoChunk& chunk) {
	ofxStableDiffusionVideoRequest request;
	request.prompt = chunk.prompt;
	request.negativePrompt = chunk.negativePrompt;
	request.width = chunk.width;
	request.height = chunk.height;
	request.frameCount = chunk.frameCount;
	request.fps = chunk.fps;
	request.sampleSteps = chunk.sampleSteps;
	request.cfgScale = chunk.cfgScale;
	request.strength = chunk.strength;
	request.seed = chunk.seed;
	applyPreset(&request, manifest.preset);
	request.width = chunk.width;
	request.height = chunk.height;
	request.frameCount = chunk.frameCount;
	request.fps = chunk.fps;
	request.sampleSteps = chunk.sampleSteps;
	request.cfgScale = chunk.cfgScale;
	request.strength = chunk.strength;
	request.seed = chunk.seed;
	return request;
}

inline std::string buildChunkOutputDirectory(
	const ofxStableDiffusionLongVideoManifest& manifest,
	const ofxStableDiffusionLongVideoChunk& chunk) {
	if (manifest.outputDirectory.empty()) {
		return chunk.id.empty() ? chunk.outputPrefix : chunk.id;
	}
	const std::string suffix = chunk.id.empty() ? chunk.outputPrefix : chunk.id;
	return joinPath(manifest.outputDirectory, suffix);
}

inline std::string buildManifestJson(
	const ofxStableDiffusionLongVideoManifest& manifest) {
	std::ostringstream output;
	output << "{\n";
	output << "  \"project_type\": \"long_video_render_manifest\",\n";
	output << "  \"project_name\": \"" << jsonEscape(manifest.projectName) << "\",\n";
	output << "  \"concept\": \"" << jsonEscape(manifest.conceptText) << "\",\n";
	output << "  \"continuity_bible\": \"" << jsonEscape(manifest.continuityBible) << "\",\n";
	output << "  \"output_directory\": \"" << jsonEscape(manifest.outputDirectory) << "\",\n";
	output << "  \"preset\": \"" << presetLabel(manifest.preset) << "\",\n";
	output << "  \"low_vram\": " << (manifest.lowVram ? "true" : "false") << ",\n";
	output << "  \"resume_enabled\": " << (manifest.resumeEnabled ? "true" : "false") << ",\n";
	output << "  \"chunks\": [\n";
	for (std::size_t i = 0; i < manifest.chunks.size(); ++i) {
		const auto& chunk = manifest.chunks[i];
		output << "    {\n";
		output << "      \"id\": \"" << jsonEscape(chunk.id) << "\",\n";
		output << "      \"title\": \"" << jsonEscape(chunk.title) << "\",\n";
		output << "      \"prompt\": \"" << jsonEscape(chunk.prompt) << "\",\n";
		output << "      \"negative_prompt\": \"" << jsonEscape(chunk.negativePrompt) << "\",\n";
		output << "      \"section_goal\": \"" << jsonEscape(chunk.sectionGoal) << "\",\n";
		output << "      \"continuity_note\": \"" << jsonEscape(chunk.continuityNote) << "\",\n";
		output << "      \"width\": " << chunk.width << ",\n";
		output << "      \"height\": " << chunk.height << ",\n";
		output << "      \"frame_count\": " << chunk.frameCount << ",\n";
		output << "      \"fps\": " << chunk.fps << ",\n";
		output << "      \"sample_steps\": " << chunk.sampleSteps << ",\n";
		output << "      \"cfg_scale\": " << chunk.cfgScale << ",\n";
		output << "      \"strength\": " << chunk.strength << ",\n";
		output << "      \"seed\": " << chunk.seed << ",\n";
		output << "      \"use_previous_last_frame\": " << (chunk.usePreviousLastFrame ? "true" : "false") << ",\n";
		output << "      \"output_prefix\": \"" << jsonEscape(chunk.outputPrefix) << "\"\n";
		output << "    }";
		if (i + 1 < manifest.chunks.size()) {
			output << ",";
		}
		output << "\n";
	}
	output << "  ]\n";
	output << "}";
	return output.str();
}

inline std::string buildPlaylistManifestJson(
	const ofxStableDiffusionLongVideoManifest& manifest,
	const std::vector<ofxStableDiffusionLongVideoChunkResult>& chunkResults) {
	std::ostringstream output;
	output << "{\n";
	output << "  \"project_name\": \"" << jsonEscape(manifest.projectName) << "\",\n";
	output << "  \"playlist\": [\n";
	for (std::size_t i = 0; i < chunkResults.size(); ++i) {
		const auto& chunk = chunkResults[i];
		output << "    {\n";
		output << "      \"chunk_id\": \"" << jsonEscape(chunk.chunkId) << "\",\n";
		output << "      \"success\": " << (chunk.success ? "true" : "false") << ",\n";
		output << "      \"clip_directory\": \"" << jsonEscape(chunk.clipDirectory) << "\",\n";
		output << "      \"metadata_path\": \"" << jsonEscape(chunk.metadataPath) << "\",\n";
		output << "      \"actual_seed\": " << chunk.actualSeed << ",\n";
		output << "      \"rendered_frame_count\": " << chunk.renderedFrameCount << "\n";
		output << "    }";
		if (i + 1 < chunkResults.size()) {
			output << ",";
		}
		output << "\n";
	}
	output << "  ]\n";
	output << "}\n";
	return output.str();
}

} // namespace ofxStableDiffusionLongVideoWorkflow
