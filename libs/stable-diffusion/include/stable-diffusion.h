#pragma once

#include "../source/include/stable-diffusion.h"

#ifdef __cplusplus

#define schedule_t scheduler_t

static constexpr sample_method_t EULER_A = EULER_A_SAMPLE_METHOD;
static constexpr sample_method_t EULER = EULER_SAMPLE_METHOD;
static constexpr sample_method_t HEUN = HEUN_SAMPLE_METHOD;
static constexpr sample_method_t DPM2 = DPM2_SAMPLE_METHOD;
static constexpr sample_method_t DPMPP2S_A = DPMPP2S_A_SAMPLE_METHOD;
static constexpr sample_method_t DPMPP2M = DPMPP2M_SAMPLE_METHOD;
static constexpr sample_method_t DPMPP2Mv2 = DPMPP2Mv2_SAMPLE_METHOD;
static constexpr sample_method_t LCM = LCM_SAMPLE_METHOD;
static constexpr sample_method_t N_SAMPLE_METHODS = SAMPLE_METHOD_COUNT;

// The legacy addon API exposed a lightweight schedule enum. Keep those names
// available and map DEFAULT to "ask upstream for the model default".
static constexpr schedule_t DEFAULT = SCHEDULER_COUNT;
static constexpr schedule_t DISCRETE = DISCRETE_SCHEDULER;
static constexpr schedule_t KARRAS = KARRAS_SCHEDULER;
static constexpr schedule_t AYS = AYS_SCHEDULER;
static constexpr schedule_t N_SCHEDULES = SCHEDULER_COUNT;

inline int32_t get_num_physical_cores() {
	return sd_get_num_physical_cores();
}

#endif
