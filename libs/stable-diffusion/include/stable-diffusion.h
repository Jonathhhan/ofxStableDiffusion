// This file is the public passthrough header for the stable-diffusion.cpp C API.
// In a full build, replace this file with (or include) the upstream stable-diffusion.h
// from the built or pre-built library. The definitions below match the current pinned
// release (master-585-44cca3d) and serve as a type-complete stub for unit tests.

#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------

enum sd_type_t {
    SD_TYPE_F32   = 0,
    SD_TYPE_F16   = 1,
    SD_TYPE_Q4_0  = 2,
    SD_TYPE_Q4_1  = 3,
    SD_TYPE_Q5_0  = 6,
    SD_TYPE_Q5_1  = 7,
    SD_TYPE_Q8_0  = 8,
    SD_TYPE_Q8_1  = 9,
    SD_TYPE_Q2_K  = 10,
    SD_TYPE_Q3_K  = 11,
    SD_TYPE_Q4_K  = 12,
    SD_TYPE_Q5_K  = 13,
    SD_TYPE_Q6_K  = 14,
    SD_TYPE_Q8_K  = 15,
    SD_TYPE_IQ2_XXS = 16,
    SD_TYPE_IQ2_XS  = 17,
    SD_TYPE_IQ3_XXS = 18,
    SD_TYPE_IQ1_S   = 19,
    SD_TYPE_IQ4_NL  = 20,
    SD_TYPE_IQ3_S   = 21,
    SD_TYPE_IQ2_S   = 22,
    SD_TYPE_IQ4_XS  = 23,
    SD_TYPE_I8    = 24,
    SD_TYPE_I16   = 25,
    SD_TYPE_I32   = 26,
    SD_TYPE_I64   = 27,
    SD_TYPE_F64   = 28,
    SD_TYPE_IQ1_M = 29,
    SD_TYPE_BF16  = 30,
    SD_TYPE_Q4_0_4_4 = 31,
    SD_TYPE_Q4_0_4_8 = 32,
    SD_TYPE_Q4_0_8_8 = 33,
    SD_TYPE_TQ1_0 = 34,
    SD_TYPE_TQ2_0 = 35,
    SD_TYPE_COUNT
};

enum rng_type_t {
    STD_DEFAULT_RNG = 0,
    CUDA_RNG        = 1,
    RNG_TYPE_COUNT
};

enum sample_method_t {
    EULER_A_SAMPLE_METHOD        = 0,
    EULER_SAMPLE_METHOD          = 1,
    HEUN_SAMPLE_METHOD           = 2,
    DPM2_SAMPLE_METHOD           = 3,
    DPMPP2S_A_SAMPLE_METHOD      = 4,
    DPMPP2M_SAMPLE_METHOD        = 5,
    DPMPP2Mv2_SAMPLE_METHOD      = 6,
    IPNDM_SAMPLE_METHOD          = 7,
    IPNDM_V_SAMPLE_METHOD        = 8,
    LCM_SAMPLE_METHOD            = 9,
    DDIM_TRAILING_SAMPLE_METHOD  = 10,
    TCD_SAMPLE_METHOD            = 11,
    RES_MULTISTEP_SAMPLE_METHOD  = 12,
    RES_2S_SAMPLE_METHOD         = 13,
    ER_SDE_SAMPLE_METHOD         = 14,
    SAMPLE_METHOD_COUNT
};

enum scheduler_t {
    DISCRETE_SCHEDULER     = 0,
    KARRAS_SCHEDULER       = 1,
    EXPONENTIAL_SCHEDULER  = 2,
    AYS_SCHEDULER          = 3,
    GITS_SCHEDULER         = 4,
    SGM_UNIFORM_SCHEDULER  = 5,
    SIMPLE_SCHEDULER       = 6,
    SMOOTHSTEP_SCHEDULER   = 7,
    KL_OPTIMAL_SCHEDULER   = 8,
    LCM_SCHEDULER          = 9,
    BONG_TANGENT_SCHEDULER = 10,
    SCHEDULER_COUNT
};

enum prediction_t {
    EPS_PRED       = 0,
    V_PRED         = 1,
    EDM_PRED       = 2,
    FLOW_PRED      = 3,
    PREDICTION_COUNT
};

enum lora_apply_mode_t {
    LORA_APPLY_AUTO    = 0,
    LORA_APPLY_TENSOR  = 1,
    LORA_APPLY_ROPE    = 2,
    LORA_APPLY_MODE_COUNT
};

enum sd_log_level_t {
    SD_LOG_DEBUG = 0,
    SD_LOG_INFO  = 1,
    SD_LOG_WARN  = 2,
    SD_LOG_ERROR = 3,
    SD_LOG_COUNT
};

enum sd_cache_mode_t {
    SD_CACHE_DISABLED    = 0,
    SD_CACHE_EASYCACHE   = 1,
    SD_CACHE_UCACHE      = 2,
    SD_CACHE_DBCACHE     = 3,
    SD_CACHE_TAYLORSEER  = 4,
    SD_CACHE_CACHE_DIT   = 5,
    SD_CACHE_SPECTRUM    = 6,
    SD_CACHE_COUNT
};

// ---------------------------------------------------------------------------
// Image
// ---------------------------------------------------------------------------

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint8_t* data;
} sd_image_t;

// ---------------------------------------------------------------------------
// Opaque context types
// ---------------------------------------------------------------------------

typedef struct sd_ctx_t sd_ctx_t;
typedef struct upscaler_ctx_t upscaler_ctx_t;

// ---------------------------------------------------------------------------
// Helper parameter structs
// ---------------------------------------------------------------------------

typedef struct {
    bool enabled;
    float target_overlap;
} sd_tiling_params_t;

typedef struct {
    float scale;
} sd_slg_params_t;

typedef struct {
    float txt_cfg;
    float img_cfg;
    float distilled_guidance;
    sd_slg_params_t slg;
} sd_guidance_params_t;

typedef struct {
    enum sample_method_t sample_method;
    enum scheduler_t scheduler;
    int sample_steps;
    sd_guidance_params_t guidance;
    float eta;
    float flow_shift;
} sd_sample_params_t;

typedef struct {
    const char* name;
    const char* path;
} sd_embedding_t;

typedef struct {
    bool is_high_noise;
    float strength;
    const char* path;
} sd_lora_t;

typedef struct {
    sd_image_t* id_images;
    int id_images_count;
    const char* id_embed_path;
    float style_strength;
} sd_pm_params_t;

// Cache parameters
typedef struct {
    enum sd_cache_mode_t mode;
    float reuse_threshold;
    float start_percent;
    float end_percent;
    float error_decay_rate;
    bool use_relative_threshold;
    bool reset_error_on_compute;
    int Fn_compute_blocks;
    int Bn_compute_blocks;
    float residual_diff_threshold;
    int max_warmup_steps;
    int max_cached_steps;
    int max_continuous_cached_steps;
    int taylorseer_n_derivatives;
    int taylorseer_skip_interval;
    const float* scm_mask;
    bool scm_policy_dynamic;
    float spectrum_w;
    int spectrum_m;
    float spectrum_lam;
    int spectrum_window_size;
    float spectrum_flex_window;
    int spectrum_warmup_steps;
    float spectrum_stop_percent;
} sd_cache_params_t;

// ---------------------------------------------------------------------------
// Context creation parameters
// ---------------------------------------------------------------------------

typedef struct {
    const char* model_path;
    const char* clip_l_path;
    const char* clip_g_path;
    const char* t5xxl_path;
    const char* diffusion_model_path;
    const char* vae_path;
    const char* taesd_path;
    const char* control_net_path;
    const char* lora_model_dir;
    const char* embed_dir;
    const char* stacked_id_embed_dir;
    const char* photo_maker_path;
    sd_embedding_t* embeddings;
    uint32_t embedding_count;
    bool vae_decode_only;
    bool vae_tiling;
    bool free_params_immediately;
    int n_threads;
    enum sd_type_t wtype;
    enum rng_type_t rng_type;
    enum rng_type_t sampler_rng_type;
    enum prediction_t prediction;
    enum lora_apply_mode_t lora_apply_mode;
    bool offload_params_to_cpu;
    bool enable_mmap;
    bool keep_clip_on_cpu;
    bool keep_control_net_on_cpu;
    bool keep_vae_on_cpu;
    bool flash_attn;
    bool diffusion_flash_attn;
    bool normalize_input;
    float style_strength;
    const char* input_id_images_path;
} sd_ctx_params_t;

// ---------------------------------------------------------------------------
// Image generation parameters
// ---------------------------------------------------------------------------

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t mask_image;
    int width;
    int height;
    sd_sample_params_t sample_params;
    float strength;
    int64_t seed;
    int batch_count;
    sd_image_t control_image;
    float control_strength;
    sd_tiling_params_t vae_tiling_params;
    sd_lora_t* loras;
    uint32_t lora_count;
    sd_pm_params_t pm_params;
} sd_img_gen_params_t;

// ---------------------------------------------------------------------------
// Video generation parameters
// ---------------------------------------------------------------------------

typedef struct {
    const char* prompt;
    const char* negative_prompt;
    int clip_skip;
    sd_image_t init_image;
    sd_image_t end_image;
    sd_image_t* control_frames;
    int control_frames_size;
    int width;
    int height;
    sd_sample_params_t sample_params;
    sd_sample_params_t high_noise_sample_params;
    float strength;
    int64_t seed;
    int video_frames;
    float moe_boundary;
    float vace_strength;
    sd_tiling_params_t vae_tiling_params;
    sd_cache_params_t cache;
    sd_lora_t* loras;
    uint32_t lora_count;
} sd_vid_gen_params_t;

// ---------------------------------------------------------------------------
// Callback types
// ---------------------------------------------------------------------------

typedef void (*sd_log_cb_t)(enum sd_log_level_t level, const char* text, void* data);
typedef bool (*sd_progress_cb_t)(int step, int steps, float time, void* data);

// ---------------------------------------------------------------------------
// Function declarations
// ---------------------------------------------------------------------------

void sd_ctx_params_init(sd_ctx_params_t* params);
void sd_img_gen_params_init(sd_img_gen_params_t* params);
void sd_vid_gen_params_init(sd_vid_gen_params_t* params);
void sd_cache_params_init(sd_cache_params_t* params);

sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* params);
void free_sd_ctx(sd_ctx_t* sd_ctx);

upscaler_ctx_t* new_upscaler_ctx(
    const char* esrgan_path,
    int n_threads,
    bool vae_tiling,
    enum sd_type_t wtype);
void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

sd_image_t* generate_image(sd_ctx_t* sd_ctx, sd_img_gen_params_t* params);
sd_image_t* generate_video(sd_ctx_t* sd_ctx, sd_vid_gen_params_t* params, int* frames_count);
sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor);

enum sample_method_t sd_get_default_sample_method(sd_ctx_t* sd_ctx);
enum scheduler_t sd_get_default_scheduler(sd_ctx_t* sd_ctx, enum sample_method_t sample_method);

const char* sd_sample_method_name(enum sample_method_t sample_method);
const char* sd_scheduler_name(enum scheduler_t scheduler);
const char* sd_rng_type_name(enum rng_type_t rng_type);
const char* sd_type_name(enum sd_type_t type);

void sd_set_log_callback(sd_log_cb_t cb, void* data);
void sd_set_progress_callback(sd_progress_cb_t cb, void* data);

const char* sd_get_system_info(void);

#ifdef __cplusplus
}
#endif
