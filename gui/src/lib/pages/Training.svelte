<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { onMount, onDestroy } from 'svelte';
  import Card from '$lib/components/Card.svelte';
  import LineChart from '$lib/components/LineChart.svelte';
  import ProgressRing from '$lib/components/ProgressRing.svelte';
  import { trainingState, datasetInfo, addActivity } from '$lib/stores/app';
  import { Play, Square, Settings } from 'lucide-svelte';

  let epochs = $state(50);
  let batchSize = $state(32);
  let learningRate = $state(0.0001);
  let labeledRatio = $state(0.2);
  let confidenceThreshold = $state(0.9);
  let dataDir = $derived($datasetInfo?.path || 'data/plantvillage/balanced');
  let outputDir = $state('output/models');

  let showSettings = $state(false);
  let unlisteners: (() => void)[] = [];

  onMount(async () => {
    const epochUnlisten = await listen<{
      epoch: number;
      total_epochs: number;
      train_loss: number;
      val_accuracy: number;
      learning_rate: number;
    }>('training:epoch', (event) => {
      trainingState.update(s => ({
        ...s,
        epoch: event.payload.epoch,
        totalEpochs: event.payload.total_epochs,
        currentLoss: event.payload.train_loss,
        currentAccuracy: event.payload.val_accuracy,
        lossHistory: [...s.lossHistory, event.payload.train_loss],
        accuracyHistory: [...s.accuracyHistory, event.payload.val_accuracy],
        learningRateHistory: [...s.learningRateHistory, event.payload.learning_rate],
      }));
    });
    unlisteners.push(epochUnlisten);

    const batchUnlisten = await listen<{
      epoch: number;
      batch: number;
      total_batches: number;
      loss: number;
    }>('training:batch', (event) => {
      trainingState.update(s => ({
        ...s,
        batch: event.payload.batch,
        totalBatches: event.payload.total_batches,
      }));
    });
    unlisteners.push(batchUnlisten);

    const completeUnlisten = await listen<{
      final_accuracy: number;
      epochs_completed: number;
      model_path: string;
    }>('training:complete', (event) => {
      trainingState.update(s => ({
        ...s,
        status: 'completed',
        currentAccuracy: event.payload.final_accuracy,
      }));
      addActivity('success', `Training complete! Final accuracy: ${event.payload.final_accuracy.toFixed(1)}%`);
    });
    unlisteners.push(completeUnlisten);
  });

  onDestroy(() => {
    unlisteners.forEach(fn => fn());
  });

  async function startTraining() {
    trainingState.update(s => ({
      ...s,
      status: 'running',
      epoch: 0,
      totalEpochs: epochs,
      batch: 0,
      totalBatches: 0,
      currentLoss: 0,
      currentAccuracy: 0,
      lossHistory: [],
      accuracyHistory: [],
      learningRateHistory: [],
    }));

    addActivity('info', 'Training started...');

    try {
      await invoke('start_training', {
        params: {
          data_dir: dataDir,
          epochs,
          batch_size: batchSize,
          learning_rate: learningRate,
          labeled_ratio: labeledRatio,
          confidence_threshold: confidenceThreshold,
          output_dir: outputDir,
        }
      });
    } catch (e) {
      trainingState.update(s => ({
        ...s,
        status: 'error',
        errorMessage: String(e),
      }));
      addActivity('error', `Training failed: ${e}`);
    }
  }

  async function stopTraining() {
    await invoke('stop_training');
    trainingState.update(s => ({
      ...s,
      status: 'idle',
    }));
    addActivity('warning', 'Training stopped');
  }

  const progress = $derived(
    $trainingState.totalEpochs > 0
      ? ($trainingState.epoch / $trainingState.totalEpochs) * 100
      : 0
  );

  const isRunning = $derived($trainingState.status === 'running');
</script>

<div class="p-6 space-y-6">
  <div class="flex items-center justify-between">
    <h2 class="text-2xl font-bold text-gray-800">Training</h2>
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
          onclick={stopTraining}
        >
          <Square class="w-4 h-4" />
          Stop
        </button>
      {:else}
        <button
          class="btn-primary flex items-center gap-2"
          onclick={startTraining}
        >
          <Play class="w-4 h-4" />
          Start Training
        </button>
      {/if}
    </div>
  </div>

  <!-- Settings Panel -->
  {#if showSettings}
    <Card title="Training Configuration">
      <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm text-gray-500 mb-1">Epochs</label>
          <input type="number" class="input w-full" bind:value={epochs} min="1" max="200" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Batch Size</label>
          <input type="number" class="input w-full" bind:value={batchSize} min="8" max="128" step="8" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Learning Rate</label>
          <input type="number" class="input w-full" bind:value={learningRate} min="0.00001" max="0.01" step="0.0001" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Labeled Ratio</label>
          <input type="number" class="input w-full" bind:value={labeledRatio} min="0.1" max="1.0" step="0.1" />
        </div>
        <div>
          <label class="block text-sm text-gray-500 mb-1">Confidence Threshold</label>
          <input type="number" class="input w-full" bind:value={confidenceThreshold} min="0.5" max="0.99" step="0.05" />
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
  <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
    <Card class="flex items-center justify-center">
      <ProgressRing value={progress} label="Progress" />
    </Card>
    
    <Card>
      <p class="text-gray-500 text-sm">Current Epoch</p>
      <p class="text-3xl font-bold text-gray-800 mt-2">
        {$trainingState.epoch}<span class="text-lg text-gray-500">/{$trainingState.totalEpochs}</span>
      </p>
      {#if $trainingState.totalBatches > 0}
        <p class="text-sm text-gray-500 mt-1">
          Batch {$trainingState.batch}/{$trainingState.totalBatches}
        </p>
      {/if}
    </Card>
    
    <Card>
      <p class="text-gray-500 text-sm">Training Loss</p>
      <p class="text-3xl font-bold text-gray-800 mt-2">
        {$trainingState.currentLoss.toFixed(4)}
      </p>
      <p class="text-sm text-gray-500 mt-1">Cross-entropy</p>
    </Card>
    
    <Card>
      <p class="text-gray-500 text-sm">Validation Accuracy</p>
      <p class="text-3xl font-bold text-blue-600 mt-2">
        {$trainingState.currentAccuracy.toFixed(1)}%
      </p>
      <p class="text-sm text-gray-500 mt-1">
        {$trainingState.status === 'completed' ? 'Final' : 'Current'}
      </p>
    </Card>
  </div>

  <!-- Charts -->
  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <Card title="Loss History">
      <div class="h-64">
        {#if $trainingState.lossHistory.length > 0}
          <LineChart 
            data={$trainingState.lossHistory}
            label="Training Loss"
            color="#EF4444"
            yAxisLabel="Loss"
          />
        {:else}
          <div class="h-full flex items-center justify-center text-gray-400">
            Start training to see loss curve
          </div>
        {/if}
      </div>
    </Card>

    <Card title="Accuracy History">
      <div class="h-64">
        {#if $trainingState.accuracyHistory.length > 0}
          <LineChart 
            data={$trainingState.accuracyHistory}
            label="Validation Accuracy"
            color="#2142f1"
            yAxisLabel="Accuracy (%)"
          />
        {:else}
          <div class="h-full flex items-center justify-center text-gray-400">
            Start training to see accuracy curve
          </div>
        {/if}
      </div>
    </Card>
  </div>
</div>
