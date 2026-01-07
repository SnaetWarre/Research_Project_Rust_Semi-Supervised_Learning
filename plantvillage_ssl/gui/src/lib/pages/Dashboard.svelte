<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { open } from '@tauri-apps/plugin-dialog';
  import Card from '$lib/components/Card.svelte';
  import ProgressRing from '$lib/components/ProgressRing.svelte';
  import { modelInfo, datasetInfo, trainingState, activityLog, addActivity } from '$lib/stores/app';
  import { 
    Database, 
    Brain, 
    Activity, 
    Upload,
    CheckCircle,
    AlertCircle,
    Info
  } from 'lucide-svelte';

  let isLoadingModel = $state(false);

  async function loadModel() {
    const selected = await open({
      title: 'Select Model File',
      filters: [{ name: 'Model', extensions: ['mpk'] }],
    });

    if (selected) {
      isLoadingModel = true;
      try {
        // Remove .mpk extension if present for Burn's load_file
        const modelPath = typeof selected === 'string' 
          ? selected.replace(/\.mpk$/, '') 
          : selected;
        
        const result = await invoke<{
          loaded: boolean;
          path: string | null;
          num_classes: number;
          input_size: number;
        }>('load_model', { modelPath });

        modelInfo.set({
          loaded: result.loaded,
          path: result.path,
          numClasses: result.num_classes,
          inputSize: result.input_size,
        });
        addActivity('success', `Model loaded successfully`);
      } catch (e) {
        addActivity('error', `Failed to load model: ${e}`);
      } finally {
        isLoadingModel = false;
      }
    }
  }

  function getActivityIcon(type: string) {
    switch (type) {
      case 'success': return CheckCircle;
      case 'warning': return AlertCircle;
      case 'error': return AlertCircle;
      default: return Info;
    }
  }

  function getActivityColor(type: string) {
    switch (type) {
      case 'success': return 'text-emerald-400';
      case 'warning': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-blue-400';
    }
  }
</script>

<div class="p-6 space-y-6">
  <div class="flex items-center justify-between">
    <h2 class="text-2xl font-bold text-white">Dashboard</h2>
    <div class="flex gap-3">
      <button
        class="btn-primary flex items-center gap-2"
        onclick={loadModel}
        disabled={isLoadingModel}
      >
        <Upload class="w-4 h-4" />
        {isLoadingModel ? 'Loading...' : 'Load Model'}
      </button>
    </div>
  </div>

  <!-- Stats Grid -->
  <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
    <!-- Dataset Card -->
    <Card>
      <div class="flex items-start justify-between">
        <div>
          <p class="text-slate-400 text-sm">Dataset</p>
          <p class="text-2xl font-bold text-white mt-1">
            {$datasetInfo ? $datasetInfo.totalSamples.toLocaleString() : '—'}
          </p>
          <p class="text-sm text-slate-400 mt-1">
            {$datasetInfo ? `${$datasetInfo.numClasses} classes` : 'Not loaded'}
          </p>
        </div>
        <div class="w-12 h-12 rounded-lg bg-blue-500/20 flex items-center justify-center">
          <Database class="w-6 h-6 text-blue-400" />
        </div>
      </div>
    </Card>

    <!-- Model Card -->
    <Card>
      <div class="flex items-start justify-between">
        <div>
          <p class="text-slate-400 text-sm">Model</p>
          <p class="text-2xl font-bold text-white mt-1">
            {$modelInfo.loaded ? 'Ready' : 'Not loaded'}
          </p>
          <p class="text-sm text-slate-400 mt-1">
            {$modelInfo.loaded ? `${$modelInfo.numClasses} classes` : 'Load a model to start'}
          </p>
        </div>
        <div class="w-12 h-12 rounded-lg {$modelInfo.loaded ? 'bg-emerald-500/20' : 'bg-slate-500/20'} flex items-center justify-center">
          <Brain class="w-6 h-6 {$modelInfo.loaded ? 'text-emerald-400' : 'text-slate-400'}" />
        </div>
      </div>
    </Card>

    <!-- Training Status Card -->
    <Card>
      <div class="flex items-start justify-between">
        <div>
          <p class="text-slate-400 text-sm">Training Status</p>
          <p class="text-2xl font-bold text-white mt-1 capitalize">
            {$trainingState.status}
          </p>
          <p class="text-sm text-slate-400 mt-1">
            {#if $trainingState.status === 'running'}
              Epoch {$trainingState.epoch}/{$trainingState.totalEpochs}
            {:else if $trainingState.status === 'completed'}
              {$trainingState.currentAccuracy.toFixed(1)}% accuracy
            {:else}
              Ready to train
            {/if}
          </p>
        </div>
        <div class="w-12 h-12 rounded-lg {$trainingState.status === 'running' ? 'bg-yellow-500/20' : $trainingState.status === 'completed' ? 'bg-emerald-500/20' : 'bg-slate-500/20'} flex items-center justify-center">
          <Activity class="w-6 h-6 {$trainingState.status === 'running' ? 'text-yellow-400 animate-pulse' : $trainingState.status === 'completed' ? 'text-emerald-400' : 'text-slate-400'}" />
        </div>
      </div>
    </Card>
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- Quick Stats -->
    {#if $datasetInfo}
      <Card title="Class Distribution (Top 10)">
        <div class="space-y-3 max-h-80 overflow-y-auto pr-2">
          {#each $datasetInfo.classNames.slice(0, 10) as className, i}
            <div class="flex items-center gap-3">
              <div class="w-24 truncate text-sm text-slate-300" title={className}>
                {className.replace(/_/g, ' ')}
              </div>
              <div class="flex-1 h-2 bg-background-lighter rounded-full overflow-hidden">
                <div
                  class="h-full bg-primary rounded-full"
                  style="width: {($datasetInfo.classCounts[i] / Math.max(...$datasetInfo.classCounts)) * 100}%"
                ></div>
              </div>
              <div class="w-12 text-right text-sm text-slate-400">
                {$datasetInfo.classCounts[i]}
              </div>
            </div>
          {/each}
        </div>
      </Card>
    {/if}

    <!-- Activity Feed -->
    <Card title="Recent Activity">
      <div class="space-y-3 max-h-80 overflow-y-auto pr-2">
        {#if $activityLog.length === 0}
          <p class="text-slate-400 text-sm">No recent activity</p>
        {:else}
          {#each $activityLog as item}
            <div class="flex items-start gap-3 py-2 border-b border-slate-700/50 last:border-0">
              <svelte:component 
                this={getActivityIcon(item.type)} 
                class="w-5 h-5 mt-0.5 {getActivityColor(item.type)}" 
              />
              <div class="flex-1 min-w-0">
                <p class="text-sm text-white">{item.message}</p>
                <p class="text-xs text-slate-500 mt-0.5">
                  {item.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          {/each}
        {/if}
      </div>
    </Card>
  </div>
</div>
