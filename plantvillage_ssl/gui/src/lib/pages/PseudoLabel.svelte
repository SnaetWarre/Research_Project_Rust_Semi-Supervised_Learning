<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import Card from '$lib/components/Card.svelte';
  import ProgressRing from '$lib/components/ProgressRing.svelte';
  import BarChart from '$lib/components/BarChart.svelte';
  import { modelInfo, datasetInfo, addActivity } from '$lib/stores/app';
  import { CheckCircle, XCircle, AlertTriangle, Play, RefreshCw } from 'lucide-svelte';

  interface PseudoLabelSample {
    image_path: string;
    predicted_class: number;
    predicted_class_name: string;
    confidence: number;
    ground_truth: number | null;
    ground_truth_name: string | null;
    is_correct: boolean | null;
    accepted: boolean;
  }

  interface PseudoLabelResults {
    samples: PseudoLabelSample[];
    total_processed: number;
    total_accepted: number;
    total_rejected: number;
    precision: number;
    acceptance_rate: number;
    class_distribution: { class_id: number; class_name: string; count: number }[];
  }

  let confidenceThreshold = $state(0.9);
  let sampleCount = $state(100);
  let results = $state<PseudoLabelResults | null>(null);
  let isLoading = $state(false);
  let sampleImages = $state<string[]>([]);

  async function loadSampleImages() {
    if (!$datasetInfo) {
      addActivity('warning', 'Please load a dataset first');
      return;
    }

    isLoading = true;
    try {
      sampleImages = await invoke<string[]>('get_sample_images', {
        dataDir: $datasetInfo.path,
        count: sampleCount,
      });
      addActivity('info', `Loaded ${sampleImages.length} sample images`);
    } catch (e) {
      addActivity('error', `Failed to load samples: ${e}`);
    } finally {
      isLoading = false;
    }
  }

  async function runDemo() {
    if (!$modelInfo.loaded) {
      addActivity('warning', 'Please load a model first');
      return;
    }

    if (sampleImages.length === 0) {
      await loadSampleImages();
      if (sampleImages.length === 0) return;
    }

    isLoading = true;
    try {
      results = await invoke<PseudoLabelResults>('run_pseudo_label_demo', {
        imagePaths: sampleImages,
        confidenceThreshold,
      });
      addActivity('success', `Pseudo-labeling demo: ${results.total_accepted}/${results.total_processed} accepted (${results.precision.toFixed(1)}% precision)`);
    } catch (e) {
      addActivity('error', `Demo failed: ${e}`);
    } finally {
      isLoading = false;
    }
  }

  function formatClassName(name: string): string {
    return name.replace(/_/g, ' ').substring(0, 25);
  }

  const acceptedSamples = $derived(results?.samples.filter(s => s.accepted) || []);
  const rejectedSamples = $derived(results?.samples.filter(s => !s.accepted) || []);
</script>

<div class="p-6 space-y-6">
  <div class="flex items-center justify-between">
    <h2 class="text-2xl font-bold text-gray-800">Pseudo-Labeling Demo</h2>
    <div class="flex gap-3">
      <button
        class="btn-secondary flex items-center gap-2"
        onclick={loadSampleImages}
        disabled={isLoading}
      >
        <RefreshCw class="w-4 h-4 {isLoading ? 'animate-spin' : ''}" />
        Load Samples
      </button>
      <button
        class="btn-primary flex items-center gap-2"
        onclick={runDemo}
        disabled={isLoading || !$modelInfo.loaded}
      >
        <Play class="w-4 h-4" />
        Run Demo
      </button>
    </div>
  </div>

  <!-- Configuration -->
  <Card title="Configuration">
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div>
        <label class="block text-sm text-gray-500 mb-2">Confidence Threshold</label>
        <div class="flex items-center gap-4">
          <input
            type="range"
            class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            bind:value={confidenceThreshold}
            min="0.5"
            max="0.99"
            step="0.01"
          />
          <span class="text-gray-800 font-medium w-16 text-right">{(confidenceThreshold * 100).toFixed(0)}%</span>
        </div>
        <p class="text-xs text-gray-400 mt-1">Higher = more precise, fewer pseudo-labels</p>
      </div>
      
      <div>
        <label class="block text-sm text-gray-500 mb-2">Sample Count</label>
        <input type="number" class="input w-full" bind:value={sampleCount} min="10" max="500" step="10" />
      </div>

      <div>
        <label class="block text-sm text-gray-500 mb-2">Samples Loaded</label>
        <p class="text-2xl font-bold text-gray-800">{sampleImages.length}</p>
      </div>
    </div>
  </Card>

  {#if results}
    <!-- Statistics -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
      <Card class="text-center">
        <ProgressRing value={results.acceptance_rate} label="Acceptance" size={100} />
      </Card>
      
      <Card class="text-center">
        <ProgressRing value={results.precision} label="Precision" size={100} />
      </Card>
      
      <Card>
        <p class="text-gray-500 text-sm">Accepted</p>
        <p class="text-3xl font-bold text-emerald-600 mt-2">{results.total_accepted}</p>
        <p class="text-sm text-gray-500">samples</p>
      </Card>
      
      <Card>
        <p class="text-gray-500 text-sm">Rejected</p>
        <p class="text-3xl font-bold text-red-600 mt-2">{results.total_rejected}</p>
        <p class="text-sm text-gray-500">samples</p>
      </Card>
    </div>

    <!-- Class Distribution -->
    {#if results.class_distribution.length > 0}
      <Card title="Pseudo-Label Class Distribution">
        <div class="h-64">
          <BarChart
            data={results.class_distribution.slice(0, 10).map(c => c.count)}
            labels={results.class_distribution.slice(0, 10).map(c => formatClassName(c.class_name))}
            label="Count"
            color="#2142f1"
          />
        </div>
      </Card>
    {/if}

    <!-- Sample Results -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Accepted Samples -->
      <Card title="Accepted Pseudo-Labels">
        <div class="space-y-2 max-h-80 overflow-y-auto pr-2">
          {#each acceptedSamples.slice(0, 20) as sample}
            <div class="flex items-center justify-between p-2 bg-gray-100 rounded-lg">
              <div class="flex items-center gap-2">
                <CheckCircle class="w-4 h-4 text-emerald-600" />
                <span class="text-sm text-gray-800 truncate max-w-[150px]" title={sample.predicted_class_name}>
                  {formatClassName(sample.predicted_class_name)}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-sm text-gray-500">{(sample.confidence * 100).toFixed(1)}%</span>
                {#if sample.is_correct !== null}
                  {#if sample.is_correct}
                    <CheckCircle class="w-4 h-4 text-emerald-600" />
                  {:else}
                    <XCircle class="w-4 h-4 text-red-600" />
                  {/if}
                {/if}
              </div>
            </div>
          {/each}
          {#if acceptedSamples.length === 0}
            <p class="text-gray-400 text-sm text-center py-4">No samples accepted at this threshold</p>
          {/if}
        </div>
      </Card>

      <!-- Rejected Samples -->
      <Card title="Rejected Samples">
        <div class="space-y-2 max-h-80 overflow-y-auto pr-2">
          {#each rejectedSamples.slice(0, 20) as sample}
            <div class="flex items-center justify-between p-2 bg-gray-100 rounded-lg">
              <div class="flex items-center gap-2">
                <XCircle class="w-4 h-4 text-red-600" />
                <span class="text-sm text-gray-800 truncate max-w-[150px]" title={sample.predicted_class_name}>
                  {formatClassName(sample.predicted_class_name)}
                </span>
              </div>
              <div class="flex items-center gap-2">
                <span class="text-sm text-gray-500">{(sample.confidence * 100).toFixed(1)}%</span>
                <AlertTriangle class="w-4 h-4 text-yellow-600" />
              </div>
            </div>
          {/each}
          {#if rejectedSamples.length === 0}
            <p class="text-gray-400 text-sm text-center py-4">All samples accepted</p>
          {/if}
        </div>
      </Card>
    </div>
  {:else}
    <Card>
      <div class="h-64 flex flex-col items-center justify-center text-gray-400">
        <p class="mb-2">Run demo to see pseudo-labeling results</p>
        <p class="text-sm">Adjust confidence threshold to see how it affects acceptance rate and precision</p>
      </div>
    </Card>
  {/if}
</div>
