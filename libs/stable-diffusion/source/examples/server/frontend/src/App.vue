<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from "vue";

import CollapsibleSection from "./components/CollapsibleSection.vue";
import ImageDropzone from "./components/ImageDropzone.vue";
import StatusChip from "./components/StatusChip.vue";
import { cancelJob, getCapabilities, getJob, submitImageJob } from "./lib/api";
import { buildRequestBody, CACHE_MODES, createBlankForm, formFromCapabilities } from "./lib/form";
import { IMAGE_INPUTS } from "./lib/image-inputs";
import { assignImageEntries, clearImageEntries, filesToImageEntries, removeImageEntry } from "./lib/images";
import { createStoredRef, normalizePollIntervalMs } from "./lib/settings";
import type { Capabilities, GenerationForm, ImageEntry, ImageTarget, Job } from "./lib/types";

const baseUrl = createStoredRef<string>("sdcpp-webui-base-url", "", (value: unknown) => String(value || ""));
const pollIntervalMs = createStoredRef<number>("sdcpp-webui-poll-interval-ms", 100, normalizePollIntervalMs);
const activeTab = ref<string>("generation");
const lightboxOpen = ref(false);
const lightboxImageSrc = ref("");
const lightboxImageAlt = ref("");
const sectionState = reactive<Record<string, boolean>>({
    sample: true,
    sampleAdvanced: false,
    guidance: true,
    guidanceAdvanced: false,
    conditioning: false,
    auxiliaryImages: false,
    lora: false,
    vaeTiling: false,
    cache: false,
});
const loadingCapabilities = ref(false);
const capabilitiesError = ref("");
const serviceOnline = ref(false);
const capabilities = ref<Capabilities | null>(null);
const currentJob = ref<Job | null>(null);
const selectedOutputIndex = ref(0);
const statusMessage = ref("");
const statusTone = ref("");
const form = reactive<GenerationForm>(createBlankForm());

let pollTimer = 0;
let elapsedTimer = 0;
const nowSeconds = ref(Date.now() / 1000);

const modelName = computed(() => {
    const model = capabilities.value?.model;
    return model?.stem || model?.name || "No model info";
});

const outputFormats = computed(() => capabilities.value?.output_formats || ["png", "jpeg"]);
const samplers = computed(() => capabilities.value?.samplers || ["default"]);
const schedulers = computed(() => capabilities.value?.schedulers || ["default"]);
const availableLoras = computed(() => capabilities.value?.loras || []);
const gridImageInputs = IMAGE_INPUTS.filter((input) => input.layout === "grid");
const fullImageInputs = IMAGE_INPUTS.filter((input) => input.layout === "full");
const queueLimit = computed(() => capabilities.value?.limits?.max_queue_size ?? "unknown");
const canCancelQueued = computed(() => Boolean(capabilities.value?.features?.cancel_queued));
const canCancelGenerating = computed(() => Boolean(capabilities.value?.features?.cancel_generating));

const currentStatus = computed(() => currentJob.value?.status || "idle");
const currentImages = computed(() => currentJob.value?.result?.images || []);
const selectedImage = computed(() => {
    if (!currentImages.value.length) {
        return null;
    }
    const index = Math.min(selectedOutputIndex.value, currentImages.value.length - 1);
    const image = currentImages.value[index];
    const format = currentJob.value?.result?.output_format || form.output_format || "png";
    return `data:image/${format};base64,${image.b64_json}`;
});

const canCancelCurrentJob = computed(() => {
    if (!currentJob.value) {
        return false;
    }
    if (currentStatus.value === "queued") {
        return canCancelQueued.value;
    }
    if (currentStatus.value === "generating") {
        return canCancelGenerating.value;
    }
    return false;
});

const isJobRunning = computed(() => {
    return currentStatus.value === "queued" || currentStatus.value === "generating";
});

const sampleSummary = computed(() => {
    const scheduler = form.sample_params.scheduler || "default";
    const method = form.sample_params.sample_method || "default";
    const steps = form.sample_params.sample_steps || 0;
    const flowShift = form.sample_params.flow_shift === "" || form.sample_params.flow_shift == null
        ? "flow auto"
        : `flow ${form.sample_params.flow_shift}`;
    return `${scheduler} · ${flowShift} · ${method} · ${steps} steps`;
});

const guidanceSummary = computed(() => {
    const cfg = form.sample_params.guidance.txt_cfg;
    const distilled = form.sample_params.guidance.distilled_guidance;
    return `cfg ${cfg} · distilled ${distilled}`;
});

const loraSummary = computed(() => {
    if (!form.lora.length) {
        return "No LoRA";
    }
    return `${form.lora.length} configured`;
});

const imageInputsSummary = computed(() => {
    const parts: string[] = [];
    if (form.init_image) parts.push("init");
    if (form.mask_image) parts.push("mask");
    if (form.control_image) parts.push("control");
    if (form.ref_images.length) parts.push(`${form.ref_images.length} refs`);
    return parts.length ? parts.join(" · ") : "No images";
});

const vaeTilingSummary = computed(() => {
    if (!form.vae_tiling_params.enabled) {
        return "Disabled";
    }
    return `${form.vae_tiling_params.tile_size_x}×${form.vae_tiling_params.tile_size_y} · overlap ${form.vae_tiling_params.target_overlap}`;
});

const cacheSummary = computed(() => {
    const mode = form.cache.mode || "disabled";
    if (mode === "disabled") {
        return "Disabled";
    }
    const option = String(form.cache.option || "").trim();
    return option ? `${mode} · ${option}` : mode;
});

const conditioningSummary = computed(() => {
    const clipSkip = formatSummaryNumber(form.clip_skip, 0);
    const strength = formatSummaryNumber(form.strength);
    const controlStrength = formatSummaryNumber(form.control_strength);
    return `clip_skip ${clipSkip} · img ${strength} · control ${controlStrength}`;
});

function setMessage(message: string, tone = ""): void {
    statusMessage.value = message;
    statusTone.value = tone;
}

function clearMessage(): void {
    setMessage("", "");
}

function deepAssign(target: Record<string, any>, ...sources: Record<string, any>[]): Record<string, any> {
    for (const source of sources) {
        if (!source) continue;
        for (const key of Object.keys(source)) {
            const sv = source[key];
            if (sv !== null && typeof sv === "object" && !Array.isArray(sv) &&
                target[key] !== null && typeof target[key] === "object" && !Array.isArray(target[key])) {
                deepAssign(target[key], sv);
            } else {
                target[key] = sv;
            }
        }
    }
    return target;
}

function applyForm(nextForm: GenerationForm): void {
    deepAssign(form, createBlankForm(), nextForm);
}

async function refreshCapabilities(): Promise<void> {
    loadingCapabilities.value = true;
    capabilitiesError.value = "";
    try {
        const response = await getCapabilities(baseUrl.value);
        capabilities.value = response;
        serviceOnline.value = true;
        applyForm(formFromCapabilities(response));
        clearMessage();
    } catch (error) {
        capabilitiesError.value = error instanceof Error ? error.message : String(error);
        serviceOnline.value = false;
        setMessage(capabilitiesError.value, "error");
    } finally {
        loadingCapabilities.value = false;
    }
}

function toggleSection(section: string): void {
    sectionState[section] = !sectionState[section];
}

function stopPolling(): void {
    if (pollTimer) {
        window.clearTimeout(pollTimer);
        pollTimer = 0;
    }
}

function stopElapsedTimer(): void {
    if (elapsedTimer) {
        window.clearInterval(elapsedTimer);
        elapsedTimer = 0;
    }
}

function startElapsedTimer(): void {
    stopElapsedTimer();
    nowSeconds.value = Date.now() / 1000;
    elapsedTimer = window.setInterval(() => {
        nowSeconds.value = Date.now() / 1000;
    }, 100);
}

async function pollJob(id: string): Promise<void> {
    stopPolling();
    try {
        currentJob.value = await getJob(baseUrl.value, id);
        serviceOnline.value = true;
        if (currentStatus.value === "queued" || currentStatus.value === "generating") {
            pollTimer = window.setTimeout(() => pollJob(id), normalizePollIntervalMs(pollIntervalMs.value));
            clearMessage();
            return;
        }
        stopElapsedTimer();
        if (currentStatus.value === "completed") {
            setMessage("Generation completed.", "success");
            return;
        }
        if (currentStatus.value === "cancelled") {
            setMessage("Job cancelled.", "error");
            return;
        }
        if (currentStatus.value === "failed") {
            setMessage(currentJob.value?.error?.message || "Generation failed.", "error");
        }
    } catch (error) {
        stopElapsedTimer();
        serviceOnline.value = false;
        setMessage(error instanceof Error ? error.message : String(error), "error");
    }
}

async function generate(): Promise<void> {
    try {
        const request = buildRequestBody(form);
        clearMessage();
        selectedOutputIndex.value = 0;
        startElapsedTimer();
        currentJob.value = await submitImageJob(baseUrl.value, request);
        await pollJob(currentJob.value.id);
    } catch (error) {
        stopElapsedTimer();
        setMessage(error instanceof Error ? error.message : String(error), "error");
    }
}

async function cancelCurrentJob(): Promise<void> {
    if (!currentJob.value?.id) {
        return;
    }
    try {
        currentJob.value = await cancelJob(baseUrl.value, currentJob.value.id);
        stopPolling();
        stopElapsedTimer();
        setMessage("Job cancelled.", "error");
    } catch (error) {
        setMessage(error instanceof Error ? error.message : String(error), "error");
    }
}

function addLora(): void {
    form.lora.push({
        path: availableLoras.value[0]?.path || "",
        multiplier: 1,
        is_high_noise: false,
    });
}

function removeLora(index: number): void {
    form.lora.splice(index, 1);
}

async function assignImages(target: ImageTarget, files: FileList): Promise<void> {
    const images = await filesToImageEntries(files);
    if (!images.length) {
        return;
    }
    assignImageEntries(form, target, images);
}

function clearImage(target: ImageTarget): void {
    clearImageEntries(form, target);
}

function getFormImage(target: ImageTarget): ImageEntry | null {
    if (target === "ref_images") return null;
    return form[target];
}

function openImageEntry(image: ImageEntry | null): void {
    openLightbox(image?.dataUrl, image?.name);
}

function removeRefImage(index: number): void {
    removeImageEntry(form, "ref_images", index);
}

function selectOutput(index: number): void {
    selectedOutputIndex.value = index;
}

function formatUnixTime(seconds: number | undefined): string {
    if (!seconds) {
        return "No job";
    }
    return new Date(seconds * 1000).toLocaleString();
}

function formatElapsed(started: number | undefined, completed: number | undefined): string {
    if (!started) {
        return "Idle";
    }
    const end = completed || nowSeconds.value;
    const total = Math.max(0, end - started);
    if (total < 60) {
        return `${total.toFixed(1)}s`;
    }
    const minutes = Math.floor(total / 60);
    const seconds = total - minutes * 60;
    return `${minutes}m ${seconds.toFixed(1)}s`;
}

function formatSummaryNumber(value: number, digits = 3): string {
    const numeric = Number(value ?? 0);
    if (!Number.isFinite(numeric)) {
        return "0";
    }
    return Number(numeric.toFixed(digits)).toString();
}

function downloadSelected(): void {
    if (!selectedImage.value) {
        return;
    }
    const link = document.createElement("a");
    link.href = selectedImage.value;
    link.download = `${currentJob.value?.id || "output"}.${currentJob.value?.result?.output_format || "png"}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
}

function openLightbox(src: string | null | undefined = selectedImage.value, alt = "Expanded image"): void {
    if (!src) {
        return;
    }
    lightboxImageSrc.value = src;
    lightboxImageAlt.value = alt;
    lightboxOpen.value = true;
}

function closeLightbox(): void {
    lightboxOpen.value = false;
    lightboxImageSrc.value = "";
    lightboxImageAlt.value = "";
}

async function onPaste(event: ClipboardEvent): Promise<void> {
    if (!event.clipboardData?.files?.length) {
        return;
    }
    await assignImages("init_image", event.clipboardData.files);
    setMessage("Pasted image into init_image.", "success");
}

onMounted(() => {
    window.addEventListener("paste", onPaste);
    refreshCapabilities();
});

onBeforeUnmount(() => {
    stopPolling();
    stopElapsedTimer();
    window.removeEventListener("paste", onPaste);
});
</script>

<template>
    <div class="shell">
        <header class="page-header panel">
            <div class="page-header__top">
                <div class="page-header__copy">
                    <div class="breadcrumb">
                        <span class="breadcrumb__org">stable-diffusion.cpp</span>
                        <span class="breadcrumb__slash">/</span>
                        <span>{{ modelName }}</span>
                    </div>
                    <h1 class="page-title">{{ modelName }}</h1>
                    <p class="page-description">
                        Native async image generation interface for the local `stable-diffusion.cpp` server.
                    </p>
                </div>
                <div class="page-header__meta">
                    <StatusChip :status="serviceOnline ? 'online' : 'offline'" :label="serviceOnline ? 'service online' : 'service unavailable'" />
                    <StatusChip :label="`queue ${queueLimit}`" />
                    <StatusChip :status="currentStatus" :label="currentStatus" />
                </div>
            </div>
            <div class="page-tabs">
                <div class="page-tabs__list">
                    <button class="page-tab" :class="{ 'page-tab--active': activeTab === 'generation' }" type="button" @click="activeTab = 'generation'">Image Generation</button>
                    <button class="page-tab" :class="{ 'page-tab--active': activeTab === 'settings' }" type="button" @click="activeTab = 'settings'">Settings</button>
                </div>
                <div class="page-tabs__actions">
                    <button class="btn-secondary" type="button" @click="refreshCapabilities" :disabled="loadingCapabilities">{{ loadingCapabilities ? "Refreshing..." : "Refresh Server Info" }}</button>
                </div>
            </div>
            <div v-if="activeTab === 'settings'" class="settings">
                <div class="settings__grid">
                    <div class="field">
                        <label>Base URL</label>
                        <input v-model="baseUrl" placeholder="Leave blank to use same origin" />
                    </div>
                    <div class="field">
                        <label>Queue Limit</label>
                        <input :value="queueLimit" readonly />
                    </div>
                    <div class="field">
                        <label>Output Formats</label>
                        <input :value="outputFormats.join(', ')" readonly />
                    </div>
                    <div class="field">
                        <label>Output Format</label>
                        <select v-model="form.output_format">
                            <option v-for="format in outputFormats" :key="format" :value="format">{{ format }}</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Output Compression</label>
                        <input v-model.number="form.output_compression" type="number" min="0" max="100" />
                    </div>
                    <div class="field">
                        <label>Job Poll Interval (ms)</label>
                        <input v-model.number="pollIntervalMs" type="number" min="1" step="1" />
                    </div>
                </div>
                <div v-if="capabilitiesError" class="status-message status-message--error">{{ capabilitiesError }}</div>
            </div>
        </header>

        <div v-if="activeTab === 'generation'" class="layout">
            <section class="panel control-panel">
                <div class="panel-header">
                    <div>
                        <h2 class="panel-title">Input</h2>
                    </div>
                </div>

                <div class="prompt-card">
                    <div class="field--full">
                        <label>Prompt</label>
                        <textarea v-model="form.prompt" placeholder="Describe the image you want to generate" />
                    </div>
                    <div class="field--full stack-top">
                        <label>Negative Prompt</label>
                        <textarea v-model="form.negative_prompt" placeholder="What should be excluded?" />
                    </div>
                </div>

                <div class="fields stack-top">
                    <div class="field"><label>Width</label><input v-model.number="form.width" type="number" min="64" /></div>
                    <div class="field"><label>Height</label><input v-model.number="form.height" type="number" min="64" /></div>
                </div>

                <div class="fields stack-top">
                    <div class="field"><label>Batch Count</label><input v-model.number="form.batch_count" type="number" min="1" /></div>
                    <div class="field"><label>Seed</label><input v-model.number="form.seed" type="number" /></div>
                </div>

                <CollapsibleSection class="stack-top" eyebrow="Sample" :summary="sampleSummary" :open="sectionState.sample" variant="module" @toggle="toggleSection('sample')">
                    <div class="fields">
                        <div class="field">
                            <label>Scheduler</label>
                            <select v-model="form.sample_params.scheduler">
                                <option value="default">default</option>
                                <option v-for="scheduler in schedulers" :key="scheduler" :value="scheduler">{{ scheduler }}</option>
                            </select>
                        </div>
                        <div class="field"><label>Flow Shift</label><input v-model="form.sample_params.flow_shift" type="number" step="0.01" placeholder="blank = default" /></div>
                        <div class="field">
                            <label>Method</label>
                            <select v-model="form.sample_params.sample_method">
                                <option value="default">default</option>
                                <option v-for="sampler in samplers" :key="sampler" :value="sampler">{{ sampler }}</option>
                            </select>
                        </div>
                        <div class="field"><label>Steps</label><input v-model.number="form.sample_params.sample_steps" type="number" /></div>
                    </div>
                    <div class="sample-panel__extras">
                        <button class="module-card__link" type="button" @click="toggleSection('sampleAdvanced')">
                            {{ sectionState.sampleAdvanced ? "Hide extras" : "Show extras" }}
                        </button>
                    </div>
                    <div v-if="sectionState.sampleAdvanced" class="fields">
                        <div class="field"><label>Eta</label><input v-model="form.sample_params.eta" type="number" step="0.01" placeholder="blank = default" /></div>
                        <div class="field"><label>Shifted Timestep</label><input v-model.number="form.sample_params.shifted_timestep" type="number" /></div>
                    </div>
                </CollapsibleSection>

                <CollapsibleSection class="stack-top" eyebrow="Guidance" :summary="guidanceSummary" :open="sectionState.guidance" variant="module" @toggle="toggleSection('guidance')">
                    <div class="fields">
                        <div class="field"><label>CFG Scale</label><input v-model.number="form.sample_params.guidance.txt_cfg" type="number" step="0.1" /></div>
                        <div class="field"><label>Distilled Guidance</label><input v-model.number="form.sample_params.guidance.distilled_guidance" type="number" step="0.1" /></div>
                    </div>
                    <div class="sample-panel__extras">
                        <button class="module-card__link" type="button" @click="toggleSection('guidanceAdvanced')">
                            {{ sectionState.guidanceAdvanced ? "Hide extras" : "Show extras" }}
                        </button>
                    </div>
                    <div v-if="sectionState.guidanceAdvanced" class="fields">
                        <div class="field"><label>Image CFG</label><input v-model="form.sample_params.guidance.img_cfg" type="number" step="0.1" placeholder="blank = follow text cfg" /></div>
                        <div class="field"><label>SLG Layers</label><input v-model="form.sample_params.guidance.slg_layers" placeholder="7,8,9" /></div>
                        <div class="field"><label>SLG Layer Start</label><input v-model.number="form.sample_params.guidance.layer_start" type="number" step="0.01" /></div>
                        <div class="field"><label>SLG Layer End</label><input v-model.number="form.sample_params.guidance.layer_end" type="number" step="0.01" /></div>
                        <div class="field"><label>SLG Scale</label><input v-model.number="form.sample_params.guidance.scale" type="number" step="0.01" /></div>
                    </div>
                </CollapsibleSection>

                <CollapsibleSection class="stack-top" eyebrow="Conditioning" :summary="conditioningSummary" :open="sectionState.conditioning" @toggle="toggleSection('conditioning')">
                    <div class="fields">
                        <div class="field"><label>CLIP Skip</label><input v-model.number="form.clip_skip" type="number" /></div>
                        <div class="field"><label>Strength</label><input v-model.number="form.strength" type="number" step="0.01" /></div>
                        <div class="field"><label>Control Strength</label><input v-model.number="form.control_strength" type="number" step="0.01" /></div>
                    </div>
                </CollapsibleSection>

                <CollapsibleSection class="stack-top" eyebrow="LoRA" :summary="loraSummary" :open="sectionState.lora" @toggle="toggleSection('lora')">
                    <div v-if="!availableLoras.length" class="hint">No LoRA entries were returned by capabilities.</div>
                    <div v-else-if="!form.lora.length" class="hint">No LoRA overrides configured.</div>
                    <div v-else class="list-editor">
                        <div class="list-row list-row--header">
                            <div>LoRA</div>
                            <div>Multiplier</div>
                            <div></div>
                        </div>
                        <div v-for="(item, index) in form.lora" :key="index" class="list-row">
                            <select v-model="item.path">
                                <option v-for="lora in availableLoras" :key="lora.path" :value="lora.path">
                                    {{ lora.name }} ({{ lora.path }})
                                </option>
                            </select>
                            <input v-model.number="item.multiplier" type="number" step="0.1" />
                            <button class="btn-ghost" type="button" @click="removeLora(index)">Remove</button>
                        </div>
                    </div>
                    <div><button class="btn-ghost" type="button" @click="addLora" :disabled="!availableLoras.length">Add LoRA</button></div>
                </CollapsibleSection>

                <CollapsibleSection class="stack-top" eyebrow="Image Inputs" :summary="imageInputsSummary" :open="sectionState.auxiliaryImages" @toggle="toggleSection('auxiliaryImages')">
                    <div class="upload-grid">
                        <ImageDropzone
                            v-for="input in gridImageInputs"
                            :key="input.target"
                            :label="input.label"
                            :description="input.description"
                            :preview="getFormImage(input.target)"
                            @select="assignImages(input.target, $event)"
                            @clear="clearImage(input.target)"
                            @preview="openImageEntry($event)"
                        />
                    </div>
                    <div v-for="input in fullImageInputs" :key="input.target">
                        <ImageDropzone
                            :label="input.label"
                            :description="input.description"
                            :preview="getFormImage(input.target)"
                            @select="assignImages(input.target, $event)"
                            @clear="clearImage(input.target)"
                            @preview="openImageEntry($event)"
                        />
                    </div>
                    <div class="group">
                        <label>Reference Images</label>
                        <ImageDropzone
                            label="Reference Images"
                            description="Multiple reference images supported."
                            :items="form.ref_images"
                            multiple
                            @select="assignImages('ref_images', $event)"
                            @clear="clearImage('ref_images')"
                        />
                        <div v-if="!form.ref_images.length" class="hint">No files selected.</div>
                        <div v-else class="file-list">
                            <div v-for="(item, index) in form.ref_images" :key="item.name + index" class="file-chip file-chip--preview">
                                <button class="file-chip__thumb-button" type="button" @click="openLightbox(item.dataUrl, item.name)">
                                    <img class="file-chip__thumb" :src="item.dataUrl" :alt="item.name" />
                                </button>
                                <span class="file-chip__name">{{ item.name }}</span>
                                <button class="icon-button" type="button" @click="removeRefImage(index)">Remove</button>
                            </div>
                        </div>
                    </div>
                </CollapsibleSection>

                <CollapsibleSection class="stack-top" eyebrow="VAE Tiling" :summary="vaeTilingSummary" :open="sectionState.vaeTiling" @toggle="toggleSection('vaeTiling')">
                    <label class="checkbox"><input v-model="form.vae_tiling_params.enabled" type="checkbox" /><span>Enabled</span></label>
                    <div class="fields">
                        <div class="field"><label>Tile Size X</label><input v-model.number="form.vae_tiling_params.tile_size_x" type="number" /></div>
                        <div class="field"><label>Tile Size Y</label><input v-model.number="form.vae_tiling_params.tile_size_y" type="number" /></div>
                    </div>
                    <div class="field"><label>Target Overlap</label><input v-model.number="form.vae_tiling_params.target_overlap" type="number" step="0.01" /></div>
                    <div class="fields">
                        <div class="field"><label>Relative Size X</label><input v-model.number="form.vae_tiling_params.rel_size_x" type="number" step="0.01" /></div>
                        <div class="field"><label>Relative Size Y</label><input v-model.number="form.vae_tiling_params.rel_size_y" type="number" step="0.01" /></div>
                    </div>
                </CollapsibleSection>

                <CollapsibleSection class="stack-top" eyebrow="Cache" :summary="cacheSummary" :open="sectionState.cache" @toggle="toggleSection('cache')">
                    <div class="field">
                        <label>Mode</label>
                        <select v-model="form.cache.mode">
                            <option v-for="mode in CACHE_MODES" :key="mode" :value="mode">{{ mode }}</option>
                        </select>
                    </div>
                    <div class="field field--full"><label>Cache Option</label><input v-model="form.cache.option" placeholder="threshold=0.25,start=0.15,end=0.95" /></div>
                    <div class="field"><label>SCM Mask</label><input v-model="form.cache.scm_mask" /></div>
                    <label class="checkbox"><input v-model="form.cache.scm_policy_dynamic" type="checkbox" /><span>Dynamic SCM Policy</span></label>
                </CollapsibleSection>
            </section>

            <section class="panel output-panel">
                <div class="panel-header">
                    <div>
                        <h2 class="panel-title">Output</h2>
                    </div>
                </div>

                <button class="hero-frame hero-frame--button" type="button" :disabled="!selectedImage" @click="openLightbox(selectedImage, 'Generated output')">
                    <img v-if="selectedImage" :src="selectedImage" alt="Generated output" />
                    <div v-else class="hero-placeholder">
                        <h2>Run a prediction</h2>
                        <p>Generated images will appear here once the current job finishes.</p>
                    </div>
                </button>

                <div class="metrics output-metrics">
                    <div class="metric">
                        <div class="metric__label">Status</div>
                        <div class="metric__value">{{ currentStatus }}</div>
                    </div>
                    <div class="metric">
                        <div class="metric__label">Queue</div>
                        <div class="metric__value">{{ currentJob?.queue_position ?? 0 }}</div>
                    </div>
                    <div class="metric">
                        <div class="metric__label">Created</div>
                        <div class="metric__value mono">{{ formatUnixTime(currentJob?.created) }}</div>
                    </div>
                    <div class="metric">
                        <div class="metric__label">Elapsed</div>
                        <div class="metric__value">{{ formatElapsed(currentJob?.started, currentJob?.completed) }}</div>
                    </div>
                </div>

                <div v-if="statusMessage" class="status-message" :class="statusTone === 'error' ? 'status-message--error' : 'status-message--success'">{{ statusMessage }}</div>
                <div v-else-if="currentJob?.error?.message" class="status-message status-message--error">{{ currentJob.error.message }}</div>

                <div class="output-controls">
                    <button class="btn output-controls__primary" type="button" :disabled="isJobRunning" @click="generate">
                        Generate
                    </button>
                    <div class="actions output-controls__secondary">
                        <button class="btn-secondary" type="button" :disabled="!selectedImage" @click="downloadSelected">Download</button>
                        <button class="btn-danger" type="button" :disabled="!canCancelCurrentJob" @click="cancelCurrentJob">Cancel</button>
                    </div>
                </div>

                <div v-if="currentImages.length > 1" class="thumb-row">
                    <button v-for="(image, index) in currentImages" :key="image.index" class="thumb" :class="{ 'thumb--active': index === selectedOutputIndex }" type="button" @click="selectOutput(index)">
                        <img :src="`data:image/${currentJob?.result?.output_format || 'png'};base64,${image.b64_json}`" :alt="`Output ${index + 1}`" />
                    </button>
                </div>
            </section>
        </div>
        <div v-if="lightboxOpen && lightboxImageSrc" class="lightbox" @click.self="closeLightbox">
            <button class="lightbox__close" type="button" @click="closeLightbox">Close</button>
            <img class="lightbox__image" :src="lightboxImageSrc" :alt="lightboxImageAlt" />
        </div>
    </div>
</template>
