#pragma once

#include "stable-diffusion.h"

#include <algorithm>
#include <cstdlib>

// Release the pixel buffer held by an sd_image_t and reset its metadata.
inline void ofxSdReleaseImage(sd_image_t& image) {
	if (image.data) {
		free(image.data);
		image.data = nullptr;
	}
	image.width = 0;
	image.height = 0;
	image.channel = 0;
}

// Release an array of sd_image_t results along with their backing buffer.
inline void ofxSdReleaseImageArray(sd_image_t* images, int count) {
	if (!images) {
		return;
	}
	const int safeCount = std::max(count, 0);
	for (int i = 0; i < safeCount; ++i) {
		ofxSdReleaseImage(images[i]);
	}
	free(images);
}

