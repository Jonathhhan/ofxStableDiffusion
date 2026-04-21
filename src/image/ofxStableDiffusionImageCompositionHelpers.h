#pragma once

#include "ofMain.h"
#include "../core/ofxStableDiffusionTypes.h"
#include <vector>
#include <cmath>

// Blend Modes for Image Composition
enum class ofxStableDiffusionBlendMode {
	Linear = 0,     // Simple linear interpolation
	Smoothstep,     // Smooth Hermite interpolation
	Cosine,         // Cosine interpolation
	Cubic,          // Cubic interpolation
	Overlay,        // Photoshop-like overlay
	Screen,         // Screen blend mode
	Multiply        // Multiply blend mode
};

// Grid Layout Configuration
struct ofxStableDiffusionGridLayout {
	int columns = 3;
	int rows = 3;
	int cellWidth = 512;
	int cellHeight = 512;
	int padding = 4;
	ofColor backgroundColor = ofColor(32, 32, 32);
	bool drawBorders = true;
	ofColor borderColor = ofColor(128, 128, 128);
	int borderWidth = 2;
	bool addLabels = false;
	int labelHeight = 24;
	ofColor labelColor = ofColor(255, 255, 255);
};

// Comparison Grid Configuration
struct ofxStableDiffusionComparisonConfig {
	std::vector<std::string> labels;
	bool showParameters = true;
	bool showSeeds = true;
	int fontSize = 12;
};

namespace ofxStableDiffusionImageCompositionHelpers {

// Blend two pixels using specified blend mode
inline ofColor blendPixels(const ofColor& a, const ofColor& b, float t, ofxStableDiffusionBlendMode mode) {
	switch (mode) {
	case ofxStableDiffusionBlendMode::Linear:
		return ofColor(
			static_cast<unsigned char>(a.r * (1.0f - t) + b.r * t),
			static_cast<unsigned char>(a.g * (1.0f - t) + b.g * t),
			static_cast<unsigned char>(a.b * (1.0f - t) + b.b * t),
			static_cast<unsigned char>(a.a * (1.0f - t) + b.a * t)
		);

	case ofxStableDiffusionBlendMode::Smoothstep: {
		float smooth_t = t * t * (3.0f - 2.0f * t);
		return ofColor(
			static_cast<unsigned char>(a.r * (1.0f - smooth_t) + b.r * smooth_t),
			static_cast<unsigned char>(a.g * (1.0f - smooth_t) + b.g * smooth_t),
			static_cast<unsigned char>(a.b * (1.0f - smooth_t) + b.b * smooth_t),
			static_cast<unsigned char>(a.a * (1.0f - smooth_t) + b.a * smooth_t)
		);
	}

	case ofxStableDiffusionBlendMode::Cosine: {
		float cos_t = (1.0f - std::cos(t * 3.14159265f)) * 0.5f;
		return ofColor(
			static_cast<unsigned char>(a.r * (1.0f - cos_t) + b.r * cos_t),
			static_cast<unsigned char>(a.g * (1.0f - cos_t) + b.g * cos_t),
			static_cast<unsigned char>(a.b * (1.0f - cos_t) + b.b * cos_t),
			static_cast<unsigned char>(a.a * (1.0f - cos_t) + b.a * cos_t)
		);
	}

	case ofxStableDiffusionBlendMode::Overlay: {
		auto overlayChannel = [](unsigned char base, unsigned char blend) -> unsigned char {
			float b = base / 255.0f;
			float bl = blend / 255.0f;
			float result = (b < 0.5f) ? (2.0f * b * bl) : (1.0f - 2.0f * (1.0f - b) * (1.0f - bl));
			return static_cast<unsigned char>(result * 255.0f);
		};
		return ofColor(
			overlayChannel(a.r, b.r),
			overlayChannel(a.g, b.g),
			overlayChannel(a.b, b.b),
			static_cast<unsigned char>(a.a * (1.0f - t) + b.a * t)
		);
	}

	case ofxStableDiffusionBlendMode::Screen: {
		auto screenChannel = [](unsigned char a, unsigned char b) -> unsigned char {
			float result = 1.0f - (1.0f - a / 255.0f) * (1.0f - b / 255.0f);
			return static_cast<unsigned char>(result * 255.0f);
		};
		return ofColor(
			screenChannel(a.r, b.r),
			screenChannel(a.g, b.g),
			screenChannel(a.b, b.b),
			static_cast<unsigned char>(a.a * (1.0f - t) + b.a * t)
		);
	}

	case ofxStableDiffusionBlendMode::Multiply: {
		return ofColor(
			static_cast<unsigned char>((a.r / 255.0f) * (b.r / 255.0f) * 255.0f),
			static_cast<unsigned char>((a.g / 255.0f) * (b.g / 255.0f) * 255.0f),
			static_cast<unsigned char>((a.b / 255.0f) * (b.b / 255.0f) * 255.0f),
			static_cast<unsigned char>(a.a * (1.0f - t) + b.a * t)
		);
	}

	default:
		return blendPixels(a, b, t, ofxStableDiffusionBlendMode::Linear);
	}
}

// Blend two images
inline bool blendImages(
	const ofPixels& imageA,
	const ofPixels& imageB,
	ofPixels& output,
	float blend = 0.5f,
	ofxStableDiffusionBlendMode mode = ofxStableDiffusionBlendMode::Linear) {

	if (!imageA.isAllocated() || !imageB.isAllocated()) return false;
	if (imageA.getWidth() != imageB.getWidth() || imageA.getHeight() != imageB.getHeight()) return false;

	output.allocate(imageA.getWidth(), imageA.getHeight(), imageA.getNumChannels());

	for (size_t i = 0; i < imageA.getWidth(); ++i) {
		for (size_t j = 0; j < imageA.getHeight(); ++j) {
			ofColor colorA = imageA.getColor(i, j);
			ofColor colorB = imageB.getColor(i, j);
			ofColor blended = blendPixels(colorA, colorB, blend, mode);
			output.setColor(i, j, blended);
		}
	}

	return true;
}

// Create comparison grid from multiple images
inline bool createComparisonGrid(
	const std::vector<ofxStableDiffusionImageFrame>& frames,
	ofPixels& output,
	const ofxStableDiffusionGridLayout& layout,
	const ofxStableDiffusionComparisonConfig& config) {

	if (frames.empty()) return false;

	int totalImages = static_cast<int>(frames.size());
	int cols = layout.columns;
	int rows = (totalImages + cols - 1) / cols;  // Ceiling division

	int labelHeight = config.addLabels ? layout.labelHeight : 0;
	int cellTotalHeight = layout.cellHeight + labelHeight;

	int gridWidth = cols * layout.cellWidth + (cols + 1) * layout.padding;
	int gridHeight = rows * cellTotalHeight + (rows + 1) * layout.padding;

	output.allocate(gridWidth, gridHeight, OF_PIXELS_RGB);

	// Fill background
	output.setColor(layout.backgroundColor);

	// Place each image
	for (int i = 0; i < totalImages; ++i) {
		int col = i % cols;
		int row = i / cols;

		int x = col * layout.cellWidth + (col + 1) * layout.padding;
		int y = row * cellTotalHeight + (row + 1) * layout.padding;

		if (!frames[i].isAllocated()) continue;

		// Scale/crop image to fit cell
		ofPixels scaled;
		scaled.allocate(layout.cellWidth, layout.cellHeight, frames[i].channels());

		float scaleX = static_cast<float>(layout.cellWidth) / frames[i].width();
		float scaleY = static_cast<float>(layout.cellHeight) / frames[i].height();
		float scale = std::max(scaleX, scaleY);

		int scaledW = static_cast<int>(frames[i].width() * scale);
		int scaledH = static_cast<int>(frames[i].height() * scale);
		int offsetX = (scaledW - layout.cellWidth) / 2;
		int offsetY = (scaledH - layout.cellHeight) / 2;

		// Simple nearest-neighbor scaling
		for (int py = 0; py < layout.cellHeight; ++py) {
			for (int px = 0; px < layout.cellWidth; ++px) {
				int srcX = static_cast<int>((px + offsetX) / scale);
				int srcY = static_cast<int>((py + offsetY) / scale);

				srcX = std::max(0, std::min(srcX, frames[i].width() - 1));
				srcY = std::max(0, std::min(srcY, frames[i].height() - 1));

				ofColor color = frames[i].pixels.getColor(srcX, srcY);
				scaled.setColor(px, py, color);
			}
		}

		// Copy to grid
		for (int py = 0; py < layout.cellHeight; ++py) {
			for (int px = 0; px < layout.cellWidth; ++px) {
				int gridX = x + px;
				int gridY = y + py;
				if (gridX < gridWidth && gridY < gridHeight) {
					ofColor color = scaled.getColor(px, py);
					output.setColor(gridX, gridY, color);
				}
			}
		}

		// Draw border
		if (layout.drawBorders) {
			for (int bw = 0; bw < layout.borderWidth; ++bw) {
				// Top and bottom
				for (int px = 0; px < layout.cellWidth; ++px) {
					output.setColor(x + px, y + bw, layout.borderColor);
					output.setColor(x + px, y + layout.cellHeight - 1 - bw, layout.borderColor);
				}
				// Left and right
				for (int py = 0; py < layout.cellHeight; ++py) {
					output.setColor(x + bw, y + py, layout.borderColor);
					output.setColor(x + layout.cellWidth - 1 - bw, y + py, layout.borderColor);
				}
			}
		}

		// Label area (simplified - just fill with dark background for text)
		if (config.addLabels && i < static_cast<int>(config.labels.size())) {
			int labelY = y + layout.cellHeight;
			for (int py = 0; py < labelHeight; ++py) {
				for (int px = 0; px < layout.cellWidth; ++px) {
					output.setColor(x + px, labelY + py, ofColor(0, 0, 0));
				}
			}
		}
	}

	return true;
}

// Create seed exploration grid
inline std::vector<ofxStableDiffusionImageRequest> createSeedExplorationRequests(
	const ofxStableDiffusionImageRequest& baseRequest,
	const std::vector<int64_t>& seeds) {

	std::vector<ofxStableDiffusionImageRequest> requests;

	for (int64_t seed : seeds) {
		ofxStableDiffusionImageRequest req = baseRequest;
		req.seed = seed;
		req.batchCount = 1;  // Single image per request
		requests.push_back(req);
	}

	return requests;
}

// Create parameter sweep requests
inline std::vector<ofxStableDiffusionImageRequest> createParameterSweepRequests(
	const ofxStableDiffusionImageRequest& baseRequest,
	const std::string& parameterName,
	float minValue,
	float maxValue,
	int steps) {

	std::vector<ofxStableDiffusionImageRequest> requests;

	for (int i = 0; i < steps; ++i) {
		float t = (steps > 1) ? (static_cast<float>(i) / (steps - 1)) : 0.0f;
		float value = minValue + t * (maxValue - minValue);

		ofxStableDiffusionImageRequest req = baseRequest;
		req.batchCount = 1;

		if (parameterName == "cfgScale") {
			req.cfgScale = value;
		} else if (parameterName == "steps") {
			req.sampleSteps = static_cast<int>(value);
		} else if (parameterName == "strength") {
			req.strength = value;
		}

		requests.push_back(req);
	}

	return requests;
}

// Interpolate between two images (morphing)
inline bool interpolateImages(
	const ofPixels& startImage,
	const ofPixels& endImage,
	std::vector<ofPixels>& output,
	int frameCount,
	ofxStableDiffusionBlendMode mode = ofxStableDiffusionBlendMode::Smoothstep) {

	if (!startImage.isAllocated() || !endImage.isAllocated()) return false;
	if (startImage.getWidth() != endImage.getWidth() ||
		startImage.getHeight() != endImage.getHeight()) return false;

	output.clear();
	output.resize(frameCount);

	for (int i = 0; i < frameCount; ++i) {
		float t = (frameCount > 1) ? (static_cast<float>(i) / (frameCount - 1)) : 0.0f;
		blendImages(startImage, endImage, output[i], t, mode);
	}

	return true;
}

// Calculate image difference/similarity
inline float calculateImageSimilarity(const ofPixels& imageA, const ofPixels& imageB) {
	if (!imageA.isAllocated() || !imageB.isAllocated()) return 0.0f;
	if (imageA.getWidth() != imageB.getWidth() ||
		imageA.getHeight() != imageB.getHeight()) return 0.0f;

	uint64_t totalDiff = 0;
	size_t pixelCount = imageA.getWidth() * imageA.getHeight();

	for (size_t i = 0; i < imageA.getWidth(); ++i) {
		for (size_t j = 0; j < imageA.getHeight(); ++j) {
			ofColor colorA = imageA.getColor(i, j);
			ofColor colorB = imageB.getColor(i, j);

			int diffR = std::abs(static_cast<int>(colorA.r) - static_cast<int>(colorB.r));
			int diffG = std::abs(static_cast<int>(colorA.g) - static_cast<int>(colorB.g));
			int diffB = std::abs(static_cast<int>(colorA.b) - static_cast<int>(colorB.b));

			totalDiff += (diffR + diffG + diffB);
		}
	}

	float maxPossibleDiff = pixelCount * 3 * 255.0f;
	float similarity = 1.0f - (totalDiff / maxPossibleDiff);
	return similarity;
}

// Create A/B comparison (side-by-side)
inline bool createABComparison(
	const ofPixels& imageA,
	const ofPixels& imageB,
	ofPixels& output,
	int padding = 8) {

	if (!imageA.isAllocated() || !imageB.isAllocated()) return false;

	int maxHeight = std::max(imageA.getHeight(), imageB.getHeight());
	int totalWidth = imageA.getWidth() + imageB.getWidth() + padding;

	output.allocate(totalWidth, maxHeight, OF_PIXELS_RGB);
	output.setColor(ofColor(32, 32, 32));

	// Copy image A
	for (size_t y = 0; y < imageA.getHeight(); ++y) {
		for (size_t x = 0; x < imageA.getWidth(); ++x) {
			output.setColor(x, y, imageA.getColor(x, y));
		}
	}

	// Copy image B
	int offsetX = imageA.getWidth() + padding;
	for (size_t y = 0; y < imageB.getHeight(); ++y) {
		for (size_t x = 0; x < imageB.getWidth(); ++x) {
			output.setColor(offsetX + x, y, imageB.getColor(x, y));
		}
	}

	return true;
}

} // namespace ofxStableDiffusionImageCompositionHelpers
