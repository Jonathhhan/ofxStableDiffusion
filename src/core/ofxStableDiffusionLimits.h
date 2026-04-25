#pragma once

/// @file ofxStableDiffusionLimits.h
/// @brief Validation limits and constants for ofxStableDiffusion
///
/// This header exposes all validation limits used throughout the addon,
/// allowing programmatic access to constraints for UI building, validation,
/// and documentation purposes.

namespace ofxStableDiffusionLimits {

	/// @name Dimension Constraints
	/// @{

	/// Minimum width/height for generated images (pixels)
	constexpr int MIN_DIMENSION = 1;

	/// Maximum width/height for generated images (pixels)
	constexpr int MAX_DIMENSION = 2048;

	/// @}

	/// @name Batch Processing Constraints
	/// @{

	/// Minimum batch count for image generation
	constexpr int MIN_BATCH_COUNT = 1;

	/// Maximum batch count for image generation
	constexpr int MAX_BATCH_COUNT = 16;

	/// @}

	/// @name Sampling Constraints
	/// @{

	/// Minimum number of sampling steps
	constexpr int MIN_SAMPLE_STEPS = 1;

	/// Maximum number of sampling steps
	constexpr int MAX_SAMPLE_STEPS = 200;

	/// @}

	/// @name Scale Parameters
	/// @{

	/// Minimum CFG scale (guidance strength)
	constexpr float MIN_CFG_SCALE = 0.0f;

	/// Maximum CFG scale (guidance strength)
	constexpr float MAX_CFG_SCALE = 50.0f;

	/// Minimum control strength for ControlNet
	constexpr float MIN_CONTROL_STRENGTH = 0.0f;

	/// Maximum control strength for ControlNet
	constexpr float MAX_CONTROL_STRENGTH = 2.0f;

	/// Minimum style strength
	constexpr float MIN_STYLE_STRENGTH = 0.0f;

	/// Maximum style strength
	constexpr float MAX_STYLE_STRENGTH = 100.0f;

	/// @}

	/// @name Unit Interval Parameters (0.0 - 1.0)
	/// @{

	/// Minimum value for unit interval parameters (strength, vaceStrength)
	constexpr float MIN_UNIT_INTERVAL = 0.0f;

	/// Maximum value for unit interval parameters (strength, vaceStrength)
	constexpr float MAX_UNIT_INTERVAL = 1.0f;

	/// @}

	/// @name Special Values
	/// @{

	/// Clip skip: -1 means auto (use model default)
	constexpr int CLIP_SKIP_AUTO = -1;

	/// Minimum clip skip value (when not auto)
	constexpr int MIN_CLIP_SKIP = 0;

	/// Maximum clip skip value
	constexpr int MAX_CLIP_SKIP = 12;

	/// Seed: -1 means random generation
	constexpr int64_t SEED_RANDOM = -1;

	/// Minimum seed value (when not random)
	constexpr int64_t MIN_SEED = 0;

	/// @}

	/// @name History Limits
	/// @{

	/// Maximum number of errors stored in error history
	constexpr std::size_t MAX_ERROR_HISTORY = 10;

	/// Maximum number of seeds stored in seed history
	constexpr std::size_t MAX_SEED_HISTORY = 20;

	/// @}

	/// @brief Check if a dimension is valid
	/// @param dimension Width or height in pixels
	/// @return true if dimension is within valid range
	inline constexpr bool isValidDimension(int dimension) {
		return dimension >= MIN_DIMENSION && dimension <= MAX_DIMENSION;
	}

	/// @brief Check if batch count is valid
	/// @param batchCount Number of images to generate
	/// @return true if batch count is within valid range
	inline constexpr bool isValidBatchCount(int batchCount) {
		return batchCount >= MIN_BATCH_COUNT && batchCount <= MAX_BATCH_COUNT;
	}

	/// @brief Check if sample steps count is valid
	/// @param steps Number of diffusion steps
	/// @return true if steps count is within valid range
	inline constexpr bool isValidSampleSteps(int steps) {
		return steps >= MIN_SAMPLE_STEPS && steps <= MAX_SAMPLE_STEPS;
	}

	/// @brief Check if CFG scale is valid
	/// @param scale Guidance scale value
	/// @return true if scale is within valid range
	inline constexpr bool isValidCfgScale(float scale) {
		return scale > MIN_CFG_SCALE && scale <= MAX_CFG_SCALE;
	}

	/// @brief Check if control strength is valid
	/// @param strength ControlNet strength value
	/// @return true if strength is within valid range
	inline constexpr bool isValidControlStrength(float strength) {
		return strength >= MIN_CONTROL_STRENGTH && strength <= MAX_CONTROL_STRENGTH;
	}

	/// @brief Check if style strength is valid
	/// @param strength Style strength value
	/// @return true if strength is within valid range
	inline constexpr bool isValidStyleStrength(float strength) {
		return strength >= MIN_STYLE_STRENGTH && strength <= MAX_STYLE_STRENGTH;
	}

	/// @brief Check if value is within unit interval
	/// @param value Parameter value
	/// @return true if value is between 0.0 and 1.0
	inline constexpr bool isValidUnitInterval(float value) {
		return value >= MIN_UNIT_INTERVAL && value <= MAX_UNIT_INTERVAL;
	}

	/// @brief Check if clip skip value is valid
	/// @param clipSkip Clip skip value (-1 for auto, or 0-12)
	/// @return true if clip skip is valid
	inline constexpr bool isValidClipSkip(int clipSkip) {
		return clipSkip == CLIP_SKIP_AUTO ||
			(clipSkip >= MIN_CLIP_SKIP && clipSkip <= MAX_CLIP_SKIP);
	}

	/// @brief Check if seed value is valid
	/// @param seed Seed value (-1 for random, or non-negative)
	/// @return true if seed is valid
	inline constexpr bool isValidSeed(int64_t seed) {
		return seed == SEED_RANDOM || seed >= MIN_SEED;
	}

} // namespace ofxStableDiffusionLimits
