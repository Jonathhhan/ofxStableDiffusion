import type { GenerationForm } from "./types";

function parseNumber(value: unknown, fallback: number): number {
    if (value === "" || value == null) {
        return fallback;
    }
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function nullable(value: unknown): unknown {
    return value === "" || value == null ? null : value;
}

interface SampleParamsRequest {
    sample_steps: number;
    shifted_timestep: number;
    custom_sigmas: number[];
    guidance: {
        txt_cfg: number;
        distilled_guidance: number;
        slg: {
            layers: number[];
            layer_start: number;
            layer_end: number;
            scale: number;
        };
        img_cfg?: number;
    };
    eta?: number;
    flow_shift?: number;
    scheduler?: string;
    sample_method?: string;
}

function buildSampleParams(form: GenerationForm): SampleParamsRequest {
    const eta = nullable(form.sample_params.eta);
    const flowShift = nullable(form.sample_params.flow_shift);
    const imgCfg = nullable(form.sample_params.guidance.img_cfg);

    const sampleParams: SampleParamsRequest = {
        sample_steps: parseNumber(form.sample_params.sample_steps, 20),
        shifted_timestep: parseNumber(form.sample_params.shifted_timestep, 0),
        custom_sigmas: [],
        guidance: {
            txt_cfg: parseNumber(form.sample_params.guidance.txt_cfg, 7),
            distilled_guidance: parseNumber(form.sample_params.guidance.distilled_guidance, 3.5),
            slg: {
                layers: String(form.sample_params.guidance.slg_layers || "")
                    .split(",")
                    .map((item) => Number(item.trim()))
                    .filter((value) => Number.isInteger(value)),
                layer_start: parseNumber(form.sample_params.guidance.layer_start, 0.01),
                layer_end: parseNumber(form.sample_params.guidance.layer_end, 0.2),
                scale: parseNumber(form.sample_params.guidance.scale, 0),
            },
        },
    };

    if (eta != null) {
        sampleParams.eta = Number(eta);
    }
    if (flowShift != null) {
        sampleParams.flow_shift = Number(flowShift);
    }
    if (imgCfg != null) {
        sampleParams.guidance.img_cfg = Number(imgCfg);
    }

    const scheduler =
        form.sample_params.scheduler && form.sample_params.scheduler !== "default"
            ? form.sample_params.scheduler
            : undefined;
    const sampleMethod =
        form.sample_params.sample_method && form.sample_params.sample_method !== "default"
            ? form.sample_params.sample_method
            : undefined;

    if (scheduler) {
        sampleParams.scheduler = scheduler;
    }
    if (sampleMethod) {
        sampleParams.sample_method = sampleMethod;
    }

    return sampleParams;
}

function buildLoraRequest(form: GenerationForm) {
    return form.lora
        .filter((item) => String(item.path || "").trim())
        .map((item) => ({
            path: String(item.path).trim(),
            multiplier: parseNumber(item.multiplier, 1.0),
            is_high_noise: Boolean(item.is_high_noise),
        }));
}

function buildTilingRequest(form: GenerationForm) {
    return {
        enabled: Boolean(form.vae_tiling_params.enabled),
        tile_size_x: parseNumber(form.vae_tiling_params.tile_size_x, 0),
        tile_size_y: parseNumber(form.vae_tiling_params.tile_size_y, 0),
        target_overlap: parseNumber(form.vae_tiling_params.target_overlap, 0.5),
        rel_size_x: parseNumber(form.vae_tiling_params.rel_size_x, 0),
        rel_size_y: parseNumber(form.vae_tiling_params.rel_size_y, 0),
    };
}

export function buildRequestBody(form: GenerationForm) {
    const request = {
        prompt: String(form.prompt || "").trim(),
        negative_prompt: form.negative_prompt,
        clip_skip: parseNumber(form.clip_skip, -1),
        width: parseNumber(form.width, 512),
        height: parseNumber(form.height, 512),
        strength: parseNumber(form.strength, 0.75),
        seed: parseNumber(form.seed, -1),
        batch_count: parseNumber(form.batch_count, 1),
        auto_resize_ref_image: true,
        increase_ref_index: false,
        control_strength: parseNumber(form.control_strength, 0.9),
        init_image: form.init_image ? form.init_image.dataUrl : null,
        ref_images: form.ref_images.map((item) => item.dataUrl),
        mask_image: form.mask_image ? form.mask_image.dataUrl : null,
        control_image: form.control_image ? form.control_image.dataUrl : null,
        sample_params: buildSampleParams(form),
        lora: buildLoraRequest(form),
        vae_tiling_params: buildTilingRequest(form),
        cache_mode: form.cache.mode || "disabled",
        cache_option: String(form.cache.option || ""),
        scm_mask: String(form.cache.scm_mask || ""),
        scm_policy_dynamic: Boolean(form.cache.scm_policy_dynamic),
        output_format: form.output_format,
        output_compression: parseNumber(form.output_compression, 100),
    };

    if (!request.prompt) {
        throw new Error("prompt is required");
    }

    return request;
}
