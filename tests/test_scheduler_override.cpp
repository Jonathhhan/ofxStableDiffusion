#include "core/ofxStableDiffusionTypes.h"

#include <cassert>
#include <iostream>

static bool expect(bool condition, const std::string& message) {
	if (!condition) {
		std::cerr << "FAIL: " << message << std::endl;
	}
	return condition;
}

// ---------------------------------------------------------------------------
// Helpers – resolve without a real sd_ctx_t (pass nullptr and use explicit
// scheduler so sd_get_default_scheduler is never called)
// ---------------------------------------------------------------------------

static scheduler_t resolveImageScheduler(
	const ofxStableDiffusionImageRequest& request,
	const ofxStableDiffusionContextSettings& settings) {
	const sample_method_t sampleMethod =
		(request.sampleMethod == SAMPLE_METHOD_COUNT)
			? EULER_A_SAMPLE_METHOD
			: request.sampleMethod;
	const scheduler_t effectiveSchedule =
		(request.schedule != SCHEDULER_COUNT) ? request.schedule : settings.schedule;
	// Mirror the adapter logic: if still SCHEDULER_COUNT use a known default
	return (effectiveSchedule != SCHEDULER_COUNT) ? effectiveSchedule : DISCRETE_SCHEDULER;
}

static scheduler_t resolveVideoScheduler(
	const ofxStableDiffusionVideoRequest& request,
	const ofxStableDiffusionContextSettings& settings) {
	const scheduler_t effectiveSchedule =
		(request.schedule != SCHEDULER_COUNT) ? request.schedule : settings.schedule;
	return (effectiveSchedule != SCHEDULER_COUNT) ? effectiveSchedule : DISCRETE_SCHEDULER;
}

static scheduler_t resolveHighNoiseScheduler(
	const ofxStableDiffusionVideoRequest& request,
	const ofxStableDiffusionContextSettings& settings) {
	const scheduler_t effectiveHighNoiseSchedule =
		(request.highNoiseSchedule != SCHEDULER_COUNT) ? request.highNoiseSchedule
		: (request.schedule != SCHEDULER_COUNT) ? request.schedule
		: settings.schedule;
	return (effectiveHighNoiseSchedule != SCHEDULER_COUNT)
		? effectiveHighNoiseSchedule
		: DISCRETE_SCHEDULER;
}

// ---------------------------------------------------------------------------
// Image request scheduler tests
// ---------------------------------------------------------------------------

static bool testImageRequestSchedulerDefaultsToContext() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = KARRAS_SCHEDULER;

	ofxStableDiffusionImageRequest request;
	// request.schedule stays SCHEDULER_COUNT (unset)

	const scheduler_t result = resolveImageScheduler(request, settings);
	ok &= expect(result == KARRAS_SCHEDULER,
		"image request with SCHEDULER_COUNT falls back to context KARRAS");
	return ok;
}

static bool testImageRequestSchedulerOverridesContext() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = KARRAS_SCHEDULER;

	ofxStableDiffusionImageRequest request;
	request.schedule = EXPONENTIAL_SCHEDULER;

	const scheduler_t result = resolveImageScheduler(request, settings);
	ok &= expect(result == EXPONENTIAL_SCHEDULER,
		"image request EXPONENTIAL overrides context KARRAS");
	return ok;
}

static bool testImageRequestSchedulerWhenContextIsDefault() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	// settings.schedule == SCHEDULER_COUNT (unset)

	ofxStableDiffusionImageRequest request;
	request.schedule = LCM_SCHEDULER;

	const scheduler_t result = resolveImageScheduler(request, settings);
	ok &= expect(result == LCM_SCHEDULER,
		"image request LCM used when context schedule is SCHEDULER_COUNT");
	return ok;
}

// ---------------------------------------------------------------------------
// Video request scheduler tests
// ---------------------------------------------------------------------------

static bool testVideoRequestSchedulerDefaultsToContext() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = AYS_SCHEDULER;

	ofxStableDiffusionVideoRequest request;
	// request.schedule == SCHEDULER_COUNT

	const scheduler_t result = resolveVideoScheduler(request, settings);
	ok &= expect(result == AYS_SCHEDULER,
		"video request with SCHEDULER_COUNT falls back to context AYS");
	return ok;
}

static bool testVideoRequestSchedulerOverridesContext() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = AYS_SCHEDULER;

	ofxStableDiffusionVideoRequest request;
	request.schedule = GITS_SCHEDULER;

	const scheduler_t result = resolveVideoScheduler(request, settings);
	ok &= expect(result == GITS_SCHEDULER,
		"video request GITS overrides context AYS");
	return ok;
}

// ---------------------------------------------------------------------------
// Video high-noise scheduler tests
// ---------------------------------------------------------------------------

static bool testHighNoiseSchedulerDefaultsToContext() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = KARRAS_SCHEDULER;

	ofxStableDiffusionVideoRequest request;
	request.useHighNoiseOverrides = true;
	// highNoiseSchedule == SCHEDULER_COUNT, schedule == SCHEDULER_COUNT

	const scheduler_t result = resolveHighNoiseScheduler(request, settings);
	ok &= expect(result == KARRAS_SCHEDULER,
		"high-noise with nothing set falls back to context KARRAS");
	return ok;
}

static bool testHighNoiseSchedulerInheritsRequestScheduler() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = KARRAS_SCHEDULER;

	ofxStableDiffusionVideoRequest request;
	request.useHighNoiseOverrides = true;
	request.schedule = EXPONENTIAL_SCHEDULER;
	// highNoiseSchedule == SCHEDULER_COUNT – should inherit request.schedule

	const scheduler_t result = resolveHighNoiseScheduler(request, settings);
	ok &= expect(result == EXPONENTIAL_SCHEDULER,
		"high-noise inherits request.schedule when highNoiseSchedule is unset");
	return ok;
}

static bool testHighNoiseSchedulerOverridesRequestScheduler() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = KARRAS_SCHEDULER;

	ofxStableDiffusionVideoRequest request;
	request.useHighNoiseOverrides = true;
	request.schedule = EXPONENTIAL_SCHEDULER;
	request.highNoiseSchedule = SMOOTHSTEP_SCHEDULER;

	const scheduler_t result = resolveHighNoiseScheduler(request, settings);
	ok &= expect(result == SMOOTHSTEP_SCHEDULER,
		"highNoiseSchedule SMOOTHSTEP overrides request EXPONENTIAL and context KARRAS");
	return ok;
}

static bool testHighNoiseSchedulerOverridesContextWhenRequestIsDefault() {
	bool ok = true;
	ofxStableDiffusionContextSettings settings;
	settings.schedule = KARRAS_SCHEDULER;

	ofxStableDiffusionVideoRequest request;
	request.useHighNoiseOverrides = true;
	// request.schedule == SCHEDULER_COUNT
	request.highNoiseSchedule = LCM_SCHEDULER;

	const scheduler_t result = resolveHighNoiseScheduler(request, settings);
	ok &= expect(result == LCM_SCHEDULER,
		"highNoiseSchedule LCM used even when request.schedule is unset");
	return ok;
}

// ---------------------------------------------------------------------------
// Type-completeness checks: verify both request types have the new fields
// ---------------------------------------------------------------------------

static bool testImageRequestHasScheduleField() {
	ofxStableDiffusionImageRequest request;
	const bool ok = request.schedule == SCHEDULER_COUNT;
	if (!ok) {
		std::cerr << "FAIL: ofxStableDiffusionImageRequest.schedule default is not SCHEDULER_COUNT"
			<< std::endl;
	}
	return ok;
}

static bool testVideoRequestHasScheduleFields() {
	ofxStableDiffusionVideoRequest request;
	bool ok = true;
	ok &= expect(request.schedule == SCHEDULER_COUNT,
		"VideoRequest.schedule defaults to SCHEDULER_COUNT");
	ok &= expect(request.highNoiseSchedule == SCHEDULER_COUNT,
		"VideoRequest.highNoiseSchedule defaults to SCHEDULER_COUNT");
	return ok;
}

int main() {
	bool ok = true;

	ok &= testImageRequestHasScheduleField();
	ok &= testVideoRequestHasScheduleFields();

	ok &= testImageRequestSchedulerDefaultsToContext();
	ok &= testImageRequestSchedulerOverridesContext();
	ok &= testImageRequestSchedulerWhenContextIsDefault();

	ok &= testVideoRequestSchedulerDefaultsToContext();
	ok &= testVideoRequestSchedulerOverridesContext();

	ok &= testHighNoiseSchedulerDefaultsToContext();
	ok &= testHighNoiseSchedulerInheritsRequestScheduler();
	ok &= testHighNoiseSchedulerOverridesRequestScheduler();
	ok &= testHighNoiseSchedulerOverridesContextWhenRequestIsDefault();

	if (!ok) {
		return 1;
	}
	std::cout << "Scheduler override tests passed" << std::endl;
	return 0;
}
