import { createBlankForm } from "./form-defaults";
import type { Capabilities, GenerationForm } from "./types";

function finiteOrFallback(value: unknown, fallback: number): number {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function formFromCapabilities(capabilities: Capabilities): GenerationForm {
    const defaults: Record<string, any> = capabilities && capabilities.defaults ? capabilities.defaults : {};
    const sample = defaults.sample_params || {};
    const guidance = sample.guidance || {};
    const slg = guidance.slg || {};
    const tiling = defaults.vae_tiling_params || {};
    const form = createBlankForm();

    form.prompt = defaults.prompt || "";
    form.negative_prompt = defaults.negative_prompt || "";
    form.width = finiteOrFallback(defaults.width, 512);
    form.height = finiteOrFallback(defaults.height, 512);
    form.batch_count = finiteOrFallback(defaults.batch_count, 1);
    form.seed = typeof defaults.seed === "number" ? defaults.seed : -1;
    form.clip_skip = typeof defaults.clip_skip === "number" ? defaults.clip_skip : -1;
    form.strength = finiteOrFallback(defaults.strength, 0.75);
    form.control_strength = finiteOrFallback(defaults.control_strength, 0.9);
    form.output_format = defaults.output_format || "png";
    form.output_compression = finiteOrFallback(defaults.output_compression, 100);

    form.sample_params.scheduler = sample.scheduler || "default";
    form.sample_params.sample_method = sample.sample_method || "default";
    form.sample_params.sample_steps = finiteOrFallback(sample.sample_steps, 20);
    form.sample_params.eta = sample.eta == null ? "" : sample.eta;
    form.sample_params.shifted_timestep = finiteOrFallback(sample.shifted_timestep, 0);
    form.sample_params.flow_shift = sample.flow_shift == null ? "" : sample.flow_shift;

    form.sample_params.guidance.txt_cfg = finiteOrFallback(guidance.txt_cfg, 7);
    form.sample_params.guidance.img_cfg = guidance.img_cfg == null ? "" : guidance.img_cfg;
    form.sample_params.guidance.distilled_guidance = finiteOrFallback(guidance.distilled_guidance, 3.5);
    form.sample_params.guidance.slg_layers = Array.isArray(slg.layers) ? slg.layers.join(",") : "7,8,9";
    form.sample_params.guidance.layer_start = finiteOrFallback(slg.layer_start, 0.01);
    form.sample_params.guidance.layer_end = finiteOrFallback(slg.layer_end, 0.2);
    form.sample_params.guidance.scale = finiteOrFallback(slg.scale, 0);

    form.vae_tiling_params.enabled = Boolean(tiling.enabled);
    form.vae_tiling_params.tile_size_x = finiteOrFallback(tiling.tile_size_x, 0);
    form.vae_tiling_params.tile_size_y = finiteOrFallback(tiling.tile_size_y, 0);
    form.vae_tiling_params.target_overlap = finiteOrFallback(tiling.target_overlap, 0.5);
    form.vae_tiling_params.rel_size_x = finiteOrFallback(tiling.rel_size_x, 0);
    form.vae_tiling_params.rel_size_y = finiteOrFallback(tiling.rel_size_y, 0);

    form.cache.mode = defaults.cache_mode || "disabled";
    form.cache.option = defaults.cache_option || "";
    form.cache.scm_mask = defaults.scm_mask || "";
    form.cache.scm_policy_dynamic =
        typeof defaults.scm_policy_dynamic === "boolean"
            ? defaults.scm_policy_dynamic
            : true;

    return form;
}
