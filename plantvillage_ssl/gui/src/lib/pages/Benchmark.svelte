<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import Card from '$lib/components/Card.svelte';
  import BarChart from '$lib/components/BarChart.svelte';
  import { modelInfo, addActivity } from '$lib/stores/app';
  import { Play, Cpu, Clock, Zap, HardDrive, Settings } from 'lucide-svelte';

  interface BenchmarkResults {
    mean_latency_ms: number;
    std_latency_ms: number;
    min_latency_ms: number;
    max_latency_ms: number;
    p50_latency_ms: number;
    p95_latency_ms: number;
    p99_latency_ms: number;
    throughput_fps: number;
    gpu_memory_mb: number | null;
    device_name: string;
    iterations: number;
    batch_size: number;
    image_size: number;
  }

  let iterations = $state(100);
  let warmup = $state(10);
  let batchSize = $state(1);
  let imageSize = $state(128);

  let results = $state<BenchmarkResults | null>(null);
  let isLoading = $state(false);
  let showSettings = $state(false);

  async function runBenchmark() {
    isLoading = true;
    try {
      results = await invoke<BenchmarkResults>('run_benchmark', {
        params: {
          model_path: $modelInfo.path?.replace(/\.mpk$/, '') || null,
          iterations,
          warmup,
          batch_size: batchSize,
          image_size: imageSize,
        }
      });
      addActivity('success', `Benchmark complete: ${results.mean_latency_ms.toFixed(2)}ms mean latency`);
    } catch (e) {
      addActivity('error', `Benchmark failed: ${e}`);
    } finally {
      isLoading = false;
    }
  }

  const latencyData = $derived(results ? [
    results.min_latency_ms,
    results.p50_latency_ms,
    results.mean_latency_ms,
    results.p95_latency_ms,
    results.p99_latency_ms,
    results.max_latency_ms,
  ] : []);

  const latencyLabels = ['Min', 'P50', 'Mean', 'P95', 'P99', 'Max'];
</script>

<div class="p-6 space-y-6">
  <div class="flex items-center justify-between">
    <h2 class="text-2xl font-bold text-white">Benchmark</h2>
    <div class="flex gap-3">
      <button
        class="btn-secondary flex items-center gap-2"
        onclick={() => showSettings = !showSettings}
      >
        <Settings class="w-4 h-4" />
        Settings
      </button>
      <button
        class="btn-primary flex items-center gap-2"
        onclick={runBenchmark}
        disabled={isLoading}
      >
        {#if isLoading}
          <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
          Running...
        {:else}
          <Play class="w-4 h-4" />
          Run Benchmark
        {/if}
      </button>
    </div>
  </div>

  <!-- Settings Panel -->
  {#if showSettings}
    <Card title="Benchmark Configuration">
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div>
          <label class="block text-sm text-slate-400 mb-1">Iterations</label>
          <input type="number" class="input w-full" bind:value={iterations} min="10" max="1000" />
        </div>
        <div>
          <label class="block text-sm text-slate-400 mb-1">Warmup</label>
          <input type="number" class="input w-full" bind:value={warmup} min="0" max="50" />
        </div>
        <div>
          <label class="block text-sm text-slate-400 mb-1">Batch Size</label>
          <input type="number" class="input w-full" bind:value={batchSize} min="1" max="32" />
        </div>
        <div>
          <label class="block text-sm text-slate-400 mb-1">Image Size</label>
          <input type="number" class="input w-full" bind:value={imageSize} min="64" max="512" />
        </div>
      </div>
    </Card>
  {/if}

  {#if results}
    <!-- Device Info -->
    <Card>
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
          <Cpu class="w-5 h-5 text-blue-400" />
        </div>
        <div>
          <p class="text-white font-medium">{results.device_name}</p>
          <p class="text-sm text-slate-400">
            {results.iterations} iterations @ batch size {results.batch_size}
          </p>
        </div>
      </div>
    </Card>

    <!-- Key Metrics -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
      <Card>
        <div class="flex items-center gap-3 mb-2">
          <Clock class="w-5 h-5 text-yellow-400" />
          <p class="text-slate-400 text-sm">Mean Latency</p>
        </div>
        <p class="text-3xl font-bold text-white">{results.mean_latency_ms.toFixed(2)}<span class="text-lg text-slate-400">ms</span></p>
        <p class="text-sm text-slate-400 mt-1">&#177; {results.std_latency_ms.toFixed(2)}ms</p>
      </Card>

      <Card>
        <div class="flex items-center gap-3 mb-2">
          <Zap class="w-5 h-5 text-emerald-400" />
          <p class="text-slate-400 text-sm">Throughput</p>
        </div>
        <p class="text-3xl font-bold text-primary">{results.throughput_fps.toFixed(1)}<span class="text-lg text-slate-400">fps</span></p>
        <p class="text-sm text-slate-400 mt-1">Images per second</p>
      </Card>

      <Card>
        <div class="flex items-center gap-3 mb-2">
          <Clock class="w-5 h-5 text-blue-400" />
          <p class="text-slate-400 text-sm">P95 Latency</p>
        </div>
        <p class="text-3xl font-bold text-white">{results.p95_latency_ms.toFixed(2)}<span class="text-lg text-slate-400">ms</span></p>
        <p class="text-sm text-slate-400 mt-1">95th percentile</p>
      </Card>

      <Card>
        <div class="flex items-center gap-3 mb-2">
          <HardDrive class="w-5 h-5 text-purple-400" />
          <p class="text-slate-400 text-sm">GPU Memory</p>
        </div>
        <p class="text-3xl font-bold text-white">
          {results.gpu_memory_mb ? `${results.gpu_memory_mb.toFixed(0)}` : '—'}<span class="text-lg text-slate-400">MB</span>
        </p>
        <p class="text-sm text-slate-400 mt-1">Used memory</p>
      </Card>
    </div>

    <!-- Latency Distribution -->
    <Card title="Latency Distribution">
      <div class="h-72">
        <BarChart
          data={latencyData}
          labels={latencyLabels}
          label="Latency (ms)"
          color="#10B981"
        />
      </div>
    </Card>

    <!-- Detailed Stats -->
    <Card title="Detailed Statistics">
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="border-b border-slate-700">
              <th class="text-left py-2 text-slate-400">Metric</th>
              <th class="text-right py-2 text-slate-400">Value</th>
            </tr>
          </thead>
          <tbody>
            <tr class="border-b border-slate-700/50">
              <td class="py-2 text-white">Minimum Latency</td>
              <td class="py-2 text-right text-slate-300">{results.min_latency_ms.toFixed(3)} ms</td>
            </tr>
            <tr class="border-b border-slate-700/50">
              <td class="py-2 text-white">Maximum Latency</td>
              <td class="py-2 text-right text-slate-300">{results.max_latency_ms.toFixed(3)} ms</td>
            </tr>
            <tr class="border-b border-slate-700/50">
              <td class="py-2 text-white">Mean Latency</td>
              <td class="py-2 text-right text-slate-300">{results.mean_latency_ms.toFixed(3)} ms</td>
            </tr>
            <tr class="border-b border-slate-700/50">
              <td class="py-2 text-white">Standard Deviation</td>
              <td class="py-2 text-right text-slate-300">{results.std_latency_ms.toFixed(3)} ms</td>
            </tr>
            <tr class="border-b border-slate-700/50">
              <td class="py-2 text-white">P50 (Median)</td>
              <td class="py-2 text-right text-slate-300">{results.p50_latency_ms.toFixed(3)} ms</td>
            </tr>
            <tr class="border-b border-slate-700/50">
              <td class="py-2 text-white">P95</td>
              <td class="py-2 text-right text-slate-300">{results.p95_latency_ms.toFixed(3)} ms</td>
            </tr>
            <tr class="border-b border-slate-700/50">
              <td class="py-2 text-white">P99</td>
              <td class="py-2 text-right text-slate-300">{results.p99_latency_ms.toFixed(3)} ms</td>
            </tr>
            <tr>
              <td class="py-2 text-white">Throughput</td>
              <td class="py-2 text-right text-slate-300">{results.throughput_fps.toFixed(2)} images/sec</td>
            </tr>
          </tbody>
        </table>
      </div>
    </Card>
  {:else}
    <Card>
      <div class="h-64 flex flex-col items-center justify-center text-slate-400">
        <Cpu class="w-16 h-16 text-slate-600 mb-4" />
        <p class="mb-2">Run a benchmark to measure inference performance</p>
        <p class="text-sm">Results include latency statistics and throughput metrics</p>
      </div>
    </Card>
  {/if}
</div>
