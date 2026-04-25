#pragma once

// Minimal stub header for stable-diffusion.cpp types
// This allows tests to compile without the full native library

#include <cstdint>

// Image structure
struct sd_image_t {
	uint32_t width;
	uint32_t height;
	uint32_t channel;
	uint8_t* data;
};

// Context and upscaler types (opaque pointers for tests)
typedef struct sd_ctx_t sd_ctx_t;
typedef struct upscaler_ctx_t upscaler_ctx_t;

// Weight/precision types
enum sd_type_t {
	SD_TYPE_F32 = 0,
	SD_TYPE_F16 = 1,
	SD_TYPE_Q4_0 = 2,
	SD_TYPE_Q4_1 = 3,
	SD_TYPE_Q5_0 = 4,
	SD_TYPE_Q5_1 = 5,
	SD_TYPE_Q8_0 = 6,
	SD_TYPE_Q8_1 = 7,
	SD_TYPE_Q2_K = 8,
	SD_TYPE_Q3_K = 9,
	SD_TYPE_Q4_K = 10,
	SD_TYPE_Q5_K = 11,
	SD_TYPE_Q6_K = 12,
	SD_TYPE_Q8_K = 13,
	SD_TYPE_IQ2_XXS = 14,
	SD_TYPE_IQ2_XS = 15,
	SD_TYPE_IQ3_XXS = 16,
	SD_TYPE_IQ1_S = 17,
	SD_TYPE_IQ4_NL = 18,
	SD_TYPE_IQ3_S = 19,
	SD_TYPE_IQ2_S = 20,
	SD_TYPE_IQ4_XS = 21,
	SD_TYPE_I8 = 22,
	SD_TYPE_I16 = 23,
	SD_TYPE_I32 = 24,
	SD_TYPE_I64 = 25,
	SD_TYPE_F64 = 26,
	SD_TYPE_IQ1_M = 27,
	SD_TYPE_BF16 = 28,
	SD_TYPE_COUNT
};

// Sample methods
enum sample_method_t {
	EULER_A_SAMPLE_METHOD = 0,
	EULER_SAMPLE_METHOD = 1,
	HEUN_SAMPLE_METHOD = 2,
	DPM2_SAMPLE_METHOD = 3,
	DPMPP2S_A_SAMPLE_METHOD = 4,
	DPMPP2M_SAMPLE_METHOD = 5,
	DPMPP2Mv2_SAMPLE_METHOD = 6,
	IPNDM_SAMPLE_METHOD = 7,
	IPNDM_V_SAMPLE_METHOD = 8,
	LCM_SAMPLE_METHOD = 9,
	DDIM_TRAILING_SAMPLE_METHOD = 10,
	TCD_SAMPLE_METHOD = 11,
	RES_MULTISTEP_SAMPLE_METHOD = 12,
	RES_2S_SAMPLE_METHOD = 13,
	ER_SDE_SAMPLE_METHOD = 14,
	SAMPLE_METHOD_COUNT
};

// Schedulers
enum scheduler_t {
	DISCRETE_SCHEDULER = 0,
	KARRAS_SCHEDULER = 1,
	EXPONENTIAL_SCHEDULER = 2,
	AYS_SCHEDULER = 3,
	GITS_SCHEDULER = 4,
	SGM_UNIFORM_SCHEDULER = 5,
	SIMPLE_SCHEDULER = 6,
	SMOOTHSTEP_SCHEDULER = 7,
	KL_OPTIMAL_SCHEDULER = 8,
	LCM_SCHEDULER = 9,
	BONG_TANGENT_SCHEDULER = 10,
	SCHEDULER_COUNT
};

// RNG types
enum rng_type_t {
	STD_DEFAULT_RNG = 0,
	CUDA_RNG = 1,
	CPU_RNG = 2
};

// Prediction types
enum prediction_t {
	V_PRED = 0,
	EPS_PRED = 1,
	FLOW_PRED = 2,
	PREDICTION_COUNT
};

// LoRA apply modes
enum lora_apply_mode_t {
	LORA_APPLY_AUTO = 0,
	LORA_APPLY_MANUAL = 1
};

// Log levels
enum sd_log_level_t {
	SD_LOG_DEBUG = 0,
	SD_LOG_INFO = 1,
	SD_LOG_WARN = 2,
	SD_LOG_ERROR = 3
};

// Cache modes
enum sd_cache_mode_t {
	SD_CACHE_DISABLED = 0,
	SD_CACHE_EASYCACHE = 1,
	SD_CACHE_UCACHE = 2,
	SD_CACHE_DBCACHE = 3,
	SD_CACHE_TAYLORSEER = 4,
	SD_CACHE_CACHE_DIT = 5,
	SD_CACHE_SPECTRUM = 6
};

// Cache parameters structure
struct sd_cache_params_t {
	sd_cache_mode_t cache_mode;
	float threshold;
	int window_size;
};

// Initialize cache params (stub)
inline void sd_cache_params_init(sd_cache_params_t* params) {
	if (params) {
		params->cache_mode = SD_CACHE_DISABLED;
		params->threshold = 0.0f;
		params->window_size = 0;
	}
}

// Stub functions (not implemented in tests, but declared for compilation)
inline const char* sd_type_name(sd_type_t type) { return "stub"; }
inline int32_t sd_get_num_physical_cores() { return 1; }
inline const char* sd_get_system_info() { return "stub"; }
