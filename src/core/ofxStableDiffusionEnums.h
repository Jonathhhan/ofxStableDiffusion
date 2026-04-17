#pragma once

enum class ofxStableDiffusionTask {
	None = 0,
	LoadModel,
	TextToImage,
	ImageToImage,
	InstructImage,
	ImageVariation,
	ImageRestyle,
	ImageToVideo,
	Upscale
};

enum class ofxStableDiffusionImageMode {
	TextToImage = 0,
	ImageToImage,
	InstructImage,
	Variation,
	Restyle
};

enum class ofxStableDiffusionVideoMode {
	Standard = 0,
	Loop,
	PingPong,
	Boomerang
};

enum class ofxStableDiffusionErrorCode {
	None = 0,
	ModelNotFound,
	ModelCorrupted,
	ModelLoadFailed,
	OutOfMemory,
	InvalidDimensions,
	InvalidBatchCount,
	InvalidFrameCount,
	MissingInputImage,
	GenerationFailed,
	ThreadBusy,
	UpscaleFailed,
	Unknown
};
