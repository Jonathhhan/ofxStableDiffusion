import type { CACHE_MODES } from "./form-defaults";

// ---------------------------------------------------------------------------
// Image helpers
// ---------------------------------------------------------------------------

export interface ImageEntry {
  name: string;
  type: string;
  dataUrl: string;
}

export type ImageTarget = "init_image" | "mask_image" | "control_image" | "ref_images";

export interface ImageInputConfig {
  target: ImageTarget;
  label: string;
  description: string;
  layout: "grid" | "full";
}

// ---------------------------------------------------------------------------
// Form model
// ---------------------------------------------------------------------------

export interface GuidanceParams {
  txt_cfg: number;
  img_cfg: string | number;
  distilled_guidance: number;
  slg_layers: string;
  layer_start: number;
  layer_end: number;
  scale: number;
}

export interface SampleParams {
  scheduler: string;
  sample_method: string;
  sample_steps: number;
  eta: string | number;
  shifted_timestep: number;
  flow_shift: string | number;
  guidance: GuidanceParams;
}

export interface VaeTilingParams {
  enabled: boolean;
  tile_size_x: number;
  tile_size_y: number;
  target_overlap: number;
  rel_size_x: number;
  rel_size_y: number;
}

export interface CacheParams {
  mode: string;
  option: string;
  scm_mask: string;
  scm_policy_dynamic: boolean;
}

export interface FormLoraEntry {
  path: string;
  multiplier: number;
  is_high_noise: boolean;
}

export interface GenerationForm {
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  batch_count: number;
  seed: number;
  clip_skip: number;
  strength: number;
  control_strength: number;
  output_format: string;
  output_compression: number;
  sample_params: SampleParams;
  init_image: ImageEntry | null;
  ref_images: ImageEntry[];
  mask_image: ImageEntry | null;
  control_image: ImageEntry | null;
  lora: FormLoraEntry[];
  vae_tiling_params: VaeTilingParams;
  cache: CacheParams;
}

// ---------------------------------------------------------------------------
// LoRA
// ---------------------------------------------------------------------------

export interface AvailableLora {
  name: string;
  path: string;
}

// ---------------------------------------------------------------------------
// Capabilities (API response)
// ---------------------------------------------------------------------------

export interface Capabilities {
  model?: {
    stem?: string;
    name?: string;
  };
  output_formats?: string[];
  samplers?: string[];
  schedulers?: string[];
  loras?: AvailableLora[];
  limits?: {
    max_queue_size?: number;
  };
  features?: {
    cancel_queued?: boolean;
    cancel_generating?: boolean;
  };
  defaults?: Record<string, any>;
}

// ---------------------------------------------------------------------------
// Job (API response)
// ---------------------------------------------------------------------------

export interface ImageOutput {
  index: number;
  b64_json: string;
}

export interface Job {
  id: string;
  status: string;
  queue_position?: number;
  created?: number;
  started?: number;
  completed?: number;
  result?: {
    images?: ImageOutput[];
    output_format?: string;
  };
  error?: {
    message?: string;
  };
}

// ---------------------------------------------------------------------------
// Cache mode literal union
// ---------------------------------------------------------------------------

export type CacheMode = (typeof CACHE_MODES)[number];
