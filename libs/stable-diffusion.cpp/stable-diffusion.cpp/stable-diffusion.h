#ifndef __STABLE_DIFFUSION_H__
#define __STABLE_DIFFUSION_H__

#include <memory>
#include <string>
#include <vector>

enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 (5) support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    // k-quantizations
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8,
    GGML_TYPE_I16,
    GGML_TYPE_I32,
    GGML_TYPE_COUNT,
};

enum RNGType {
    STD_DEFAULT_RNG,
    CUDA_RNG
};

enum SampleMethod {
    EULER_A,
    EULER,
    HEUN,
    DPM2,
    DPMPP2S_A,
    DPMPP2M,
    DPMPP2Mv2,
    LCM,
    N_SAMPLE_METHODS
};

enum Schedule {
    DEFAULT,
    DISCRETE,
    KARRAS,
    N_SCHEDULES
};

class StableDiffusionGGML;

class StableDiffusion {
private:
    std::shared_ptr<StableDiffusionGGML> sd;

public:
    void setup(int n_threads                = -1,
                    bool vae_decode_only         = false,
                    std::string taesd_path       = "",
                    std::string esrgan_path      = "",
                    bool free_params_immediately = false,
                    std::string lora_model_dir   = "",
                    RNGType rng_type             = STD_DEFAULT_RNG);
    bool load_from_file(const std::string& model_path,
                        const std::string& vae_path,
                        ggml_type wtype,
                        Schedule d = DEFAULT);
    std::vector<uint8_t*> txt2img(
        std::string prompt,
        std::string negative_prompt,
        float cfg_scale,
        int width,
        int height,
        SampleMethod sample_method,
        int sample_steps,
        int64_t seed,
        int batch_count);

    std::vector<uint8_t*> img2img(
        const uint8_t* init_img_data,
        std::string prompt,
        std::string negative_prompt,
        float cfg_scale,
        int width,
        int height,
        SampleMethod sample_method,
        int sample_steps,
        float strength,
        int64_t seed);
};

std::string sd_get_system_info();

#endif  // __STABLE_DIFFUSION_H__