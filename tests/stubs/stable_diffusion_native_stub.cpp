#include "core/ofxStableDiffusionNativeApi.h"

#include <limits>

void sd_cache_params_init(sd_cache_params_t* cache_params) {
    if (cache_params == nullptr) {
        return;
    }

    *cache_params = {};
    cache_params->mode = SD_CACHE_DISABLED;
    cache_params->reuse_threshold = std::numeric_limits<float>::infinity();
    cache_params->start_percent = 0.15f;
    cache_params->end_percent = 0.95f;
    cache_params->error_decay_rate = 1.0f;
    cache_params->use_relative_threshold = true;
    cache_params->reset_error_on_compute = true;
    cache_params->Fn_compute_blocks = 8;
    cache_params->Bn_compute_blocks = 0;
    cache_params->residual_diff_threshold = 0.08f;
    cache_params->max_warmup_steps = 8;
    cache_params->max_cached_steps = -1;
    cache_params->max_continuous_cached_steps = -1;
    cache_params->taylorseer_n_derivatives = 1;
    cache_params->taylorseer_skip_interval = 1;
    cache_params->scm_mask = nullptr;
    cache_params->scm_policy_dynamic = true;
    cache_params->spectrum_w = 0.40f;
    cache_params->spectrum_m = 3;
    cache_params->spectrum_lam = 1.0f;
    cache_params->spectrum_window_size = 2;
    cache_params->spectrum_flex_window = 0.50f;
    cache_params->spectrum_warmup_steps = 4;
    cache_params->spectrum_stop_percent = 0.9f;
}
