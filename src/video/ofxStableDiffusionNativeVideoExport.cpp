#include "ofxStableDiffusionNativeVideoExport.h"

#include "ofImage.h"
#include "ofUtils.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace ofxStableDiffusionNativeVideoExport {
namespace {

struct AviIndexEntry {
	uint32_t offset = 0;
	uint32_t size = 0;
};

using FilePtr = std::unique_ptr<FILE, decltype(&std::fclose)>;

void writeU32Le(FILE* file, uint32_t value) {
	std::fwrite(&value, sizeof(value), 1, file);
}

void writeU16Le(FILE* file, uint16_t value) {
	std::fwrite(&value, sizeof(value), 1, file);
}

ofImageQualityType qualityLevelFromPercent(int quality) {
	if (quality >= 90) {
		return OF_IMAGE_QUALITY_BEST;
	}
	if (quality >= 75) {
		return OF_IMAGE_QUALITY_HIGH;
	}
	if (quality >= 50) {
		return OF_IMAGE_QUALITY_MEDIUM;
	}
	if (quality >= 25) {
		return OF_IMAGE_QUALITY_LOW;
	}
	return OF_IMAGE_QUALITY_WORST;
}

bool saveAvi(const std::string& path, const ofxStableDiffusionVideoClip& clip, int quality) {
	if (clip.frames.empty()) {
		ofLogWarning("ofxStableDiffusion") << "Video export skipped because there are no frames.";
		return false;
	}
	if (clip.fps <= 0) {
		ofLogWarning("ofxStableDiffusion") << "Video export skipped because FPS is not positive.";
		return false;
	}

	const auto& firstFrame = clip.frames.front();
	if (!firstFrame.isAllocated()) {
		ofLogWarning("ofxStableDiffusion") << "Video export skipped because the first frame is not allocated.";
		return false;
	}

	const uint32_t width = static_cast<uint32_t>(firstFrame.width());
	const uint32_t height = static_cast<uint32_t>(firstFrame.height());

	for (std::size_t i = 0; i < clip.frames.size(); ++i) {
		const auto& frame = clip.frames[i];
		if (!frame.isAllocated()) {
			ofLogWarning("ofxStableDiffusion") << "Video export skipped because frame " << i << " is not allocated.";
			return false;
		}
		if (static_cast<uint32_t>(frame.width()) != width || static_cast<uint32_t>(frame.height()) != height) {
			ofLogWarning("ofxStableDiffusion") << "Video export skipped because frame sizes do not match.";
			return false;
		}
	}

	const std::string resolvedPath = ofFilePath::isAbsolute(path) ? path : ofToDataPath(path, true);
	ofFile outputFile(resolvedPath);
	const auto directory = outputFile.getEnclosingDirectory();
	if (!directory.empty()) {
		ofDirectory::createDirectory(directory, true, true);
	}

	FilePtr file(std::fopen(resolvedPath.c_str(), "wb"), &std::fclose);
	if (!file) {
		ofLogError("ofxStableDiffusion") << "Could not open video file for writing: " << resolvedPath;
		return false;
	}
	FILE* rawFile = file.get();

	std::fwrite("RIFF", 4, 1, rawFile);
	const long riffSizePos = std::ftell(rawFile);
	writeU32Le(rawFile, 0);
	std::fwrite("AVI ", 4, 1, rawFile);

	std::fwrite("LIST", 4, 1, rawFile);
	writeU32Le(rawFile, 4 + 8 + 56 + 8 + 4 + 8 + 56 + 8 + 40);
	std::fwrite("hdrl", 4, 1, rawFile);

	std::fwrite("avih", 4, 1, rawFile);
	writeU32Le(rawFile, 56);
	writeU32Le(rawFile, 1000000u / static_cast<uint32_t>(clip.fps));
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0x110);
	writeU32Le(rawFile, static_cast<uint32_t>(clip.frames.size()));
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 1);
	writeU32Le(rawFile, width * height * 3);
	writeU32Le(rawFile, width);
	writeU32Le(rawFile, height);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0);

	std::fwrite("LIST", 4, 1, rawFile);
	writeU32Le(rawFile, 4 + 8 + 56 + 8 + 40);
	std::fwrite("strl", 4, 1, rawFile);

	std::fwrite("strh", 4, 1, rawFile);
	writeU32Le(rawFile, 56);
	std::fwrite("vids", 4, 1, rawFile);
	std::fwrite("MJPG", 4, 1, rawFile);
	writeU32Le(rawFile, 0);
	writeU16Le(rawFile, 0);
	writeU16Le(rawFile, 0);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 1);
	writeU32Le(rawFile, static_cast<uint32_t>(clip.fps));
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, static_cast<uint32_t>(clip.frames.size()));
	writeU32Le(rawFile, width * height * 3);
	writeU32Le(rawFile, static_cast<uint32_t>(-1));
	writeU32Le(rawFile, 0);
	writeU16Le(rawFile, 0);
	writeU16Le(rawFile, 0);
	writeU16Le(rawFile, 0);
	writeU16Le(rawFile, 0);

	std::fwrite("strf", 4, 1, rawFile);
	writeU32Le(rawFile, 40);
	writeU32Le(rawFile, 40);
	writeU32Le(rawFile, width);
	writeU32Le(rawFile, height);
	writeU16Le(rawFile, 1);
	writeU16Le(rawFile, 24);
	std::fwrite("MJPG", 4, 1, rawFile);
	writeU32Le(rawFile, width * height * 3);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0);
	writeU32Le(rawFile, 0);

	std::fwrite("LIST", 4, 1, rawFile);
	const long moviSizePos = std::ftell(rawFile);
	writeU32Le(rawFile, 0);
	std::fwrite("movi", 4, 1, rawFile);

	std::vector<AviIndexEntry> index(clip.frames.size());
	ofBuffer jpegBuffer;
	const auto qualityLevel = qualityLevelFromPercent(quality);

	for (std::size_t i = 0; i < clip.frames.size(); ++i) {
		jpegBuffer.clear();
		if (!ofSaveImage(clip.frames[i].pixels, jpegBuffer, OF_IMAGE_FORMAT_JPEG, qualityLevel)) {
			ofLogError("ofxStableDiffusion") << "Failed to encode frame " << i << " as JPEG for AVI export.";
			return false;
		}

		std::fwrite("00dc", 4, 1, rawFile);
		writeU32Le(rawFile, static_cast<uint32_t>(jpegBuffer.size()));
		index[i].offset = static_cast<uint32_t>(std::ftell(rawFile) - 8);
		index[i].size = static_cast<uint32_t>(jpegBuffer.size());
		std::fwrite(jpegBuffer.getData(), 1, jpegBuffer.size(), rawFile);

		if ((jpegBuffer.size() & 1u) != 0u) {
			std::fputc(0, rawFile);
		}
	}

	long currentPos = std::ftell(rawFile);
	const long moviSize = currentPos - moviSizePos - 4;
	std::fseek(rawFile, moviSizePos, SEEK_SET);
	writeU32Le(rawFile, static_cast<uint32_t>(moviSize));
	std::fseek(rawFile, currentPos, SEEK_SET);

	std::fwrite("idx1", 4, 1, rawFile);
	writeU32Le(rawFile, static_cast<uint32_t>(clip.frames.size() * 16));
	for (const auto& entry : index) {
		std::fwrite("00dc", 4, 1, rawFile);
		writeU32Le(rawFile, 0x10);
		writeU32Le(rawFile, entry.offset);
		writeU32Le(rawFile, entry.size);
	}

	currentPos = std::ftell(rawFile);
	const long fileSize = currentPos - riffSizePos - 4;
	std::fseek(rawFile, riffSizePos, SEEK_SET);
	writeU32Le(rawFile, static_cast<uint32_t>(fileSize));
	std::fseek(rawFile, currentPos, SEEK_SET);

	return true;
}

} // namespace

bool isWebmExportAvailable() {
	return false;
}

bool saveWebm(const std::string& path, const ofxStableDiffusionVideoClip& clip, int quality) {
	std::string extension = ofFilePath::getFileExt(path);
	std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});

	if (extension != "avi") {
		ofLogWarning("ofxStableDiffusion")
			<< "Only AVI export is currently available from the addon export path. "
			<< "Use an .avi filename to match sd-cli's built-in fallback writer.";
		return false;
	}

	return saveAvi(path, clip, quality);
}

} // namespace ofxStableDiffusionNativeVideoExport
