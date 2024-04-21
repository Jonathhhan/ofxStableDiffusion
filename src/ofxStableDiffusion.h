#pragma once

#include "ofMain.h"
#include "../libs/stable-diffusion/include/stable-diffusion.h"

class ofxStableDiffusion {
public:

	void setup();
	void update();
	void exit();

	ofxStableDiffusion();

	virtual ~ofxStableDiffusion();

	sd_ctx_t* loadModel(const char* model_path,
		const char* vae_path,
		const char* taesd_path,
		const char* control_net_path_c_str,
		const char* lora_model_dir,
		const char* embed_dir_c_str,
		const char* stacked_id_embed_dir_c_str,
		bool vae_decode_only,
		bool vae_tiling,
		bool free_params_immediately,
		int n_threads,
		enum sd_type_t wtype,
		enum rng_type_t rng_type,
		enum schedule_t s,
		bool keep_clip_on_cpu,
		bool keep_control_net_cpu,
		bool keep_vae_on_cpu);

	void free_sd_ctx(sd_ctx_t* sdCtx);

	sd_image_t* txt2img(sd_ctx_t* sd_ctx,
		const char* prompt,
		const char* negative_prompt,
		int clip_skip,
		float cfg_scale,
		int width,
		int height,
		enum sample_method_t sample_method,
		int sample_steps,
		int64_t seed,
		int batch_count,
		const sd_image_t* control_cond,
		float control_strength,
		float style_strength,
		bool normalize_input,
		const char* input_id_images_path);

	sd_image_t* img2img(sd_ctx_t* sd_ctx,
		sd_image_t init_image,
		const char* prompt,
		const char* negative_prompt,
		int clip_skip,
		float cfg_scale,
		int width,
		int height,
		enum sample_method_t sample_method,
		int sample_steps,
		float strength,
		int64_t seed,
		int batch_count);

	sd_image_t* ofxStableDiffusion::img2vid(sd_ctx_t* sd_ctx,
		sd_image_t init_image,
		int width,
		int height,
		int video_frames,
		int motion_bucket_id,
		int fps,
		float augmentation_level,
		float min_cfg,
		float cfg_scale,
		enum sample_method_t sample_method,
		int sample_steps,
		float strength,
		int64_t seed);

	void new_upscaler_ctx(const char* esrgan_path,
		int n_threads,
		enum sd_type_t wtype);

	void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx);

	void upscale(upscaler_ctx_t* upscaler_ctx,
		sd_image_t input_image,
		uint32_t upscale_factor);

	bool convert(const char* input_path,
		const char* vae_path,
		const char* output_path,
		sd_type_t output_type);

	void preprocess_canny(uint8_t* img,
		int width,
		int height,
		float high_threshold,
		float low_threshold,
		float weak,
		float strong,
		bool inverse);
};