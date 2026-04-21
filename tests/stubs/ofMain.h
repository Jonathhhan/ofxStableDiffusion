#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

enum ofImageType {
	OF_IMAGE_GRAYSCALE = 1,
	OF_IMAGE_COLOR = 3,
	OF_IMAGE_COLOR_ALPHA = 4
};

class ofPixels {
public:
	bool isAllocated() const {
		return !storage.empty();
	}

	std::size_t getWidth() const {
		return width;
	}

	std::size_t getHeight() const {
		return height;
	}

	std::size_t getNumChannels() const {
		return channels;
	}

	unsigned char* getData() {
		return storage.empty() ? nullptr : storage.data();
	}

	const unsigned char* getData() const {
		return storage.empty() ? nullptr : storage.data();
	}

	void setFromPixels(const unsigned char* data, int pixelWidth, int pixelHeight, ofImageType type) {
		width = pixelWidth > 0 ? static_cast<std::size_t>(pixelWidth) : 0;
		height = pixelHeight > 0 ? static_cast<std::size_t>(pixelHeight) : 0;
		channels = static_cast<std::size_t>(std::max(0, static_cast<int>(type)));
		const std::size_t byteCount = width * height * channels;
		storage.resize(byteCount);
		if (data && byteCount > 0) {
			std::copy(data, data + byteCount, storage.begin());
		}
	}

private:
	std::size_t width = 0;
	std::size_t height = 0;
	std::size_t channels = 0;
	std::vector<unsigned char> storage;
};
