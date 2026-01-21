<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { onMount, onDestroy } from 'svelte';
  import Card from '$lib/components/Card.svelte';
  import LineChart from '$lib/components/LineChart.svelte';
  import ProgressRing from '$lib/components/ProgressRing.svelte';
  import { simulationState, modelInfo, datasetInfo, addActivity } from '$lib/stores/app';
  import { Play, Square, Calendar, Tags, TrendingUp, Settings } from 'lucide-svelte';

  let days = $state(30);
  let imagesPerDay = $state(50);
  let confidenceThreshold = $state(0.9);
  let retrainThreshold = $state(200);
  let dataDir = $derived($datasetInfo?.path || 'data/plantvillage/balanced');
  let modelPath = $state('output/models/plant_classifier');
  let outputDir = $state('output/simulation');

  let showSettings = $state(false);
  let unlisteners: (() => void)[] = [];

  interface SimulationResult {
    initial_accuracy: number;
    final_accuracy: number;
    improvement: number;
    days_simulated: number;
    total_pseudo_labels: number;
    pseudo_label_precision: number;
    retrain_count: number;
    accuracy_history: [number, number][];
  }

  let result = $state<SimulationResult | null>(null);

  // Derived states for chart data (fixes Svelte 5 state proxy issue)
  let accuracyHistoryData = $derived(result?.accuracy_history?.map(([_, acc]) => acc) ?? []);
  let accuracyHistoryLabels = $derived(result?.accuracy_history?.map(([day, _]) => `Day ${day}`) ?? []);

  onMount(async () => {
    const completeUnlisten = await listen<SimulationResult>('simulation:complete', (event) => {
      result = event.payload;
      simulationState.update(s => ({
        ...s,
        status: 'completed',
        accuracyHistory: event.payload.accuracy_history.map(([day, acc]) => ({ day, accuracy: acc })),
      }));
      addActivity('success', `Simulation complete! Improvement: ${event.payload.improvement.toFixed(2)}%`);
    });
    unlisteners.push(completeUnlisten);

    if ($modelInfo.path) {
      modelPath = $modelInfo.path.replace(/\.mpk$/, '');
    }
  });

  onDestroy(() => {
    unlisteners.forEach(fn => fn());
  });

  async function startSimulation() {
    simulationState.update(s => ({
      ...s,
      status: 'running',
      day: 0,
      totalDays: days,
      pseudoLabels: 0,
      currentAccuracy: 0,
      accuracyHistory: [],
    }));

    result = null;
    addActivity('info', 'Starting simulation...');

    try {
      await invoke('start_simulation', {
        params: {
          data_dir: dataDir,
          model_path: modelPath,
          days,
          images_per_day: imagesPerDay,
          confidence_threshold: confidenceThreshold,
          retrain_threshold: retrainThreshold,
          output_dir: outputDir,
        }
      });
    } catch (e) {
      simulationState.update(s => ({
        ...s,
        status: 'error',
        errorMessage: String(e),
      }));
      addActivity('error', `Simulation failed: ${e}`);
    }
  }

  async function stopSimulation() {
    await invoke('stop_simulation');
    simulationState.update(s => ({
      ...s,
      status: 'idle',
    }));
    addActivity('warning', 'Simulation stopped');
  }

  const progress = $derived(
    $simulationState.totalDays > 0
      ? ($simulationState.day / $simulationState.totalDays) * 100
      : 0
  );

  const isRunning = $derived($simulationState.status === 'running');
</script>

<div class="p-6 space-y-6">
  <div class="flex items-center justify-between">
    <h2 class="text-2xl font-bold text-gray-800">Stream Simulation</h2>
    <div class="flex gap-3">
      <button
        class="btn-secondary flex items-center gap-2"
        onclick={() => showSettings = !showSettings}
      >
        <Settings class="w-4 h-4" />
        Settings
      </button>
      {#if isRunning}
        <button
          class="bg-red-500 hover:bg-red-600 text-white font-medium py-2 px-4 rounded-lg flex items-center gap-2"
          onclick={stopSimulation}
        >
          <Square class="w-4 h-4" />
          Stop
        </button>
      {:else}
        <button
          class="btn-primary flex items-center gap-2"
          onclick={startSimulation}
        >
          <Play class="w-4 h-4" />
          Start Simulation
        </button>
      {/if}
    </div>
  </div>

  <!-- Settings Panel -->
  {#if showSettings}
    <Card title="Simulation Configuration">
      <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm text-gray-500 mb-1">Simulated Days</label>
          <input type="number" class="input w-full" bind:value={days} min="5" max="100" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Images Per Day</label>
          <input type="number" class="input w-full" bind:value={imagesPerDay} min="10" max="200" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Confidence Threshold</label>
          <input type="number" class="input w-full" bind:value={confidenceThreshold} min="0.5" max="0.99" step="0.05" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Retrain Threshold</label>
          <input type="number" class="input w-full" bind:value={retrainThreshold} min="50" max="500" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Model Path</label>
          <input type="text" class="input w-full" bind:value={modelPath} />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Data Directory</label>
          <input type="text" class="input w-full bg-gray-100 text-gray-500" value={dataDir} readonly />
          <p class="text-xs text-gray-400 mt-1">Auto-loaded from balanced dataset</p>
        </div>
      </div>
    </Card>
  {/if}

  <!-- Progress Overview -->
  <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
    <Card class="flex items-center justify-center">
      <ProgressRing value={progress} label="Progress" />
    </Card>
    
    <Card>
      <div class="flex items-center gap-3 mb-2">
        <Calendar class="w-5 h-5 text-blue-600" />
        <p class="text-gray-500 text-sm">Simulated Days</p>
      </div>
      <p class="text-3xl font-bold text-gray-800">
        {result?.days_simulated || $simulationState.day}<span class="text-lg text-gray-500">/{days}</span>
      </p>
    </Card>
    
    <Card>
      <div class="flex items-center gap-3 mb-2">
        <Tags class="w-5 h-5 text-emerald-600" />
        <p class="text-gray-500 text-sm">Pseudo-Labels</p>
      </div>
      <p class="text-3xl font-bold text-emerald-600">
        {result?.total_pseudo_labels || $simulationState.pseudoLabels}
      </p>
    </Card>
    
    <Card>
      <div class="flex items-center gap-3 mb-2">
        <TrendingUp class="w-5 h-5 text-blue-600" />
        <p class="text-gray-500 text-sm">Improvement</p>
      </div>
      <p class="text-3xl font-bold {(result?.improvement || 0) > 0 ? 'text-blue-600' : 'text-gray-400'}">
        {result ? `+${result.improvement.toFixed(2)}%` : '—'}
      </p>
    </Card>
  </div>

  {#if result}
    <!-- Results Summary -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Card>
        <p class="text-gray-500 text-sm">Initial Accuracy</p>
        <p class="text-3xl font-bold text-gray-800 mt-2">{result.initial_accuracy.toFixed(1)}%</p>
        <p class="text-sm text-gray-500 mt-1">Before SSL</p>
      </Card>
      
      <Card>
        <p class="text-gray-500 text-sm">Final Accuracy</p>
        <p class="text-3xl font-bold text-blue-600 mt-2">{result.final_accuracy.toFixed(1)}%</p>
        <p class="text-sm text-gray-500 mt-1">After SSL</p>
      </Card>
      
      <Card>
        <p class="text-gray-500 text-sm">Pseudo-Label Precision</p>
        <p class="text-3xl font-bold text-yellow-600 mt-2">{result.pseudo_label_precision.toFixed(1)}%</p>
        <p class="text-sm text-gray-500 mt-1">{result.retrain_count} retraining sessions</p>
      </Card>
    </div>

    <!-- Accuracy Timeline -->
    <Card title="Accuracy Over Time">
      <div class="h-72">
        {#if result.accuracy_history.length > 0}
          <LineChart 
            data={accuracyHistoryData}
            labels={accuracyHistoryLabels}
            label="Validation Accuracy"
            color="#2142f1"
            yAxisLabel="Accuracy (%)"
          />
        {:else}
          <div class="h-full flex items-center justify-center text-gray-400">
            No accuracy history available
          </div>
        {/if}
      </div>
    </Card>
  {:else if isRunning}
    <Card>
      <div class="h-48 flex flex-col items-center justify-center">
        <div class="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
        <p class="text-gray-800 font-medium">Simulation in progress...</p>
        <p class="text-gray-500 text-sm mt-1">This may take several minutes</p>
      </div>
    </Card>
  {:else}
    <Card>
      <div class="h-48 flex flex-col items-center justify-center text-gray-400">
        <p class="mb-2">Start a simulation to see SSL improvement over time</p>
        <p class="text-sm">The simulation processes images day-by-day, accumulating pseudo-labels</p>
        <p class="text-sm">and retraining when the threshold is reached</p>
      </div>
    </Card>
  {/if}
</div>
