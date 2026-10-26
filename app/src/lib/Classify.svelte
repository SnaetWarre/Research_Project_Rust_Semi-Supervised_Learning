<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { classify, className, loadModel, percent, type Classification, type ModelInfo } from "./api";

  let model: ModelInfo | null = $state(null);
  let preview: string | null = $state(null);
  let result: Classification | null = $state(null);
  let error: string | null = $state(null);
  let busy = $state(false);
  let dragging = $state(false);

  async function chooseModel() {
    const dir = await open({ directory: true, title: "Choose a model folder (it holds model.json)" });
    if (typeof dir !== "string") return;
    error = null;
    try {
      model = await loadModel(dir);
      result = null;
    } catch (e) {
      error = String(e);
    }
  }

  async function run(file: File | undefined) {
    if (!file || !model) return;
    if (preview) URL.revokeObjectURL(preview);
    preview = URL.createObjectURL(file);
    busy = true;
    error = null;
    try {
      result = await classify(new Uint8Array(await file.arrayBuffer()));
    } catch (e) {
      error = String(e);
      result = null;
    } finally {
      busy = false;
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    dragging = false;
    run(event.dataTransfer?.files[0]);
  }
</script>

<section class="space-y-6">
  <div class="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-panel px-5 py-4">
    <div class="min-w-0">
      <p class="text-sm text-ink-3">Model</p>
      {#if model}
        <p class="truncate text-sm" title={model.dir}>{model.dir}</p>
        <p class="text-xs text-ink-3">{model.classes} classes, {model.image_size} px input, runs on the CPU</p>
      {:else}
        <p class="text-sm text-ink-2">None loaded. Pick a folder made by <code>plantssl train</code> or <code>plantssl ssl</code>.</p>
      {/if}
    </div>
    <button class="rounded-md bg-leaf/15 px-3 py-1.5 text-sm text-leaf hover:bg-leaf/25" onclick={chooseModel}>
      {model ? "Change model" : "Choose model folder"}
    </button>
  </div>

  {#if error}
    <p role="alert" class="rounded-md border border-red-400/30 bg-red-400/10 px-4 py-2 text-sm text-red-200">{error}</p>
  {/if}

  <div class="grid gap-6 md:grid-cols-2">
    <label
      class="relative grid aspect-square cursor-pointer place-items-center overflow-hidden rounded-xl border-2 border-dashed transition-colors
        {dragging ? 'border-leaf bg-leaf/5' : 'border-line hover:border-ink-3'}
        {model ? '' : 'pointer-events-none opacity-50'}"
      ondragover={(e) => {
        e.preventDefault();
        dragging = true;
      }}
      ondragleave={() => (dragging = false)}
      ondrop={onDrop}
    >
      <input
        type="file"
        accept="image/*"
        class="sr-only"
        disabled={!model}
        onchange={(e) => run(e.currentTarget.files?.[0])}
      />
      {#if preview}
        <img src={preview} alt="Leaf to classify" class="absolute inset-0 size-full object-cover" />
      {:else}
        <span class="px-6 text-center text-sm text-ink-3">Drop a leaf photo here or click to choose one</span>
      {/if}
    </label>

    <div class="rounded-xl bg-panel p-5" aria-live="polite">
      {#if busy}
        <p class="text-sm text-ink-3">Classifying...</p>
      {:else if result}
        {@const [top, ...rest] = result.scores}
        <p class="text-sm text-ink-3">Most likely</p>
        <p class="mt-1 text-xl font-semibold">{className(top.class)}</p>
        <p class="text-3xl font-semibold tabular-nums text-ink">{percent(top.probability, 2)}</p>
        <ul class="mt-6 space-y-3">
          {#each rest as score (score.class)}
            <li>
              <div class="flex justify-between gap-3 text-sm">
                <span class="truncate text-ink-2">{className(score.class)}</span>
                <span class="tabular-nums text-ink-2">{percent(score.probability)}</span>
              </div>
              <div class="mt-1 h-1.5 rounded-full bg-line">
                <div class="h-full rounded-full bg-series" style:width="{score.probability * 100}%"></div>
              </div>
            </li>
          {/each}
        </ul>
        <p class="mt-6 text-xs text-ink-3">Inference took {result.millis.toFixed(1)} ms</p>
      {:else}
        <p class="text-sm text-ink-3">The five most likely diseases show up here.</p>
      {/if}
    </div>
  </div>
</section>
