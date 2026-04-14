<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { open } from '@tauri-apps/plugin-dialog';
  import { onMount, onDestroy } from 'svelte';
  import Card from '$lib/components/Card.svelte';
  import LineChart from '$lib/components/LineChart.svelte';
  import { demoState, modelInfo, datasetInfo, addActivity } from '$lib/stores/app';
  import { 
    Play, 
    RotateCcw, 
    Calendar, 
    Image as ImageIcon, 
    TrendingUp, 
    Zap,
    CheckCircle2,
    XCircle,
    Loader2,
    Settings,
    FolderOpen,
    Upload,
    RefreshCw
  } from 'lucide-svelte';

  // Configuration
  let dataDir = $datasetInfo?.path || 'data/plantvillage';
  let modelPath = $state('');
  let imagesPerDay = $state(100);
  let confidenceThreshold = $state(0.9);
  let retrainThreshold = $state(200);
  let labeledRatio = $state(0.2);
  let showSettings = $state(true); // Open by default

  let unlisteners: (() => void)[] = [];
  let isInitializing = $state(false);
  let isProcessing = $state(false);
  let isImportingFarmer = $state(false);
  let isRetraining = $state(false);
  let loadingMessage = $state('');
  
  let untilRetrain = $derived(retrainThreshold - $demoState.pseudoLabelsAccumulated);

  // Derived states for chart data (fixes Svelte 5 state proxy issue)
  let accuracyChartData = $derived($demoState.accuracyHistory.map(h => h.accuracy));
  let accuracyChartLabels = $derived($demoState.accuracyHistory.map(h => `Day ${h.day}`));

  // File picker for base CNN model
  async function selectBaseModel() {
    const selected = await open({
      title: 'Select Base CNN Model (not SSL-trained)',
      filters: [{ name: 'Model', extensions: ['mpk'] }],
    });
    
    if (selected && typeof selected === 'string') {
      modelPath = selected.replace(/\.mpk$/, '');
      addActivity('info', `Selected base model: ${modelPath.split('/').pop()}`);
    }
  }

  onMount(async () => {
    // Listen for progress events
    const progressUnlisten = await listen('demo:progress', (event: any) => {
      const data = event.payload;
      console.log('[Demo Progress]', data.step, data.message);
      loadingMessage = data.message;
    });
    unlisteners.push(progressUnlisten);

    // Listen for retraining started event
    const retrainingUnlisten = await listen('demo:retraining_started', (event: any) => {
      const data = event.payload;
      console.log('[Demo Retraining]', data);
      loadingMessage = `Retraining model with ${data.pseudo_labels} pseudo-labels... This takes ~1-2 minutes`;
      addActivity('info', 'Retraining started - model is learning from pseudo-labels');
    });
    unlisteners.push(retrainingUnlisten);

    const dayCompleteUnlisten = await listen('demo:day_complete', (event: any) => {
      const result = event.payload;
      console.log('[Demo Day Complete]', result);
      demoState.update(s => ({
        ...s,
        currentDay: result.day,
        imagesProcessed: s.imagesProcessed + result.images_processed_today,
        totalPseudoLabelsGenerated: s.totalPseudoLabelsGenerated + result.pseudo_labels_accepted_today,
        pseudoLabelsAccumulated: result.pseudo_labels_accumulated,
        currentAccuracy: result.current_accuracy,
        pseudoLabelPrecision: result.pseudo_label_precision,
        lastDayResult: result,
        status: 'initialized',
      }));
      
      if (result.did_retrain) {
        demoState.update(s => ({
          ...s,
          retrainingCount: s.retrainingCount + 1,
          accuracyHistory: [...s.accuracyHistory, { day: result.day, accuracy: result.current_accuracy }],
        }));
        addActivity('success', `Day ${result.day}: Retrained! New accuracy: ${result.current_accuracy.toFixed(1)}%`);
      } else {
        addActivity('info', `Day ${result.day}: ${result.pseudo_labels_accepted_today} pseudo-labels added`);
      }
      
      isProcessing = false;
      loadingMessage = '';
    });
    unlisteners.push(dayCompleteUnlisten);

    // Listen for farmer import complete event
    const farmerCompleteUnlisten = await listen('demo:farmer_complete', (event: any) => {
      const result = event.payload;
      console.log('[Demo Farmer Complete]', result);
      
      // Update state with farmer results - create a synthetic day result for display
      demoState.update(s => ({
        ...s,
        imagesProcessed: s.imagesProcessed + result.images_processed,
        totalPseudoLabelsGenerated: s.totalPseudoLabelsGenerated + result.pseudo_labels_accepted,
        pseudoLabelsAccumulated: result.pseudo_labels_accumulated,
        currentAccuracy: result.current_accuracy,
        pseudoLabelPrecision: result.pseudo_label_precision,
        lastDayResult: {
          day: s.currentDay,
          images_processed_today: result.images_processed,
          pseudo_labels_accepted_today: result.pseudo_labels_accepted,
          pseudo_labels_accumulated: result.pseudo_labels_accumulated,
          did_retrain: false,
          accuracy_before_retrain: null,
          accuracy_after_retrain: null,
          current_accuracy: result.current_accuracy,
          pseudo_label_precision: result.pseudo_label_precision,
          sample_images: result.sample_images,
          remaining_images: s.lastDayResult?.remaining_images || 0,
        },
        status: 'initialized',
      }));
      
      addActivity('success', `Farmer upload: ${result.pseudo_labels_accepted} of ${result.images_processed} images accepted as pseudo-labels`);
      isImportingFarmer = false;
      loadingMessage = '';
    });
    unlisteners.push(farmerCompleteUnlisten);

    // Listen for manual retrain complete event
    const manualRetrainUnlisten = await listen('demo:manual_retrain_complete', (event: any) => {
      const result = event.payload;
      console.log('[Demo Manual Retrain Complete]', result);
      
      demoState.update(s => ({
        ...s,
        currentAccuracy: result.accuracy_after,
        retrainingCount: s.retrainingCount + 1,
        pseudoLabelsAccumulated: 0,
        accuracyHistory: [...s.accuracyHistory, { day: s.currentDay, accuracy: result.accuracy_after }],
        status: 'initialized',
      }));
      
      const improvement = result.accuracy_after - result.accuracy_before;
      addActivity('success', `Manual retrain complete! Accuracy: ${result.accuracy_before.toFixed(1)}% → ${result.accuracy_after.toFixed(1)}% (+${improvement.toFixed(2)}%)`);
      isRetraining = false;
      loadingMessage = '';
    });
    unlisteners.push(manualRetrainUnlisten);

    if ($modelInfo.path) {
      modelPath = $modelInfo.path.replace(/\.mpk$/, '');
    }
  });

  onDestroy(() => {
    unlisteners.forEach(fn => fn());
  });

  async function initSession() {
    isInitializing = true;
    loadingMessage = 'Loading dataset and model...';
    demoState.update(s => ({...s, status: 'initialized', errorMessage: undefined}));
    addActivity('info', 'Initializing demo session...');

    try {
      const result: any = await invoke('init_demo_session', {
        config: {
          data_dir: dataDir,
          model_path: modelPath,
          images_per_day: imagesPerDay,
          confidence_threshold: confidenceThreshold,
          retrain_threshold: retrainThreshold,
          labeled_ratio: labeledRatio,
          seed: 42,
        }
      });

      demoState.update(s => ({
        ...s,
        status: 'initialized',
        currentDay: result.current_day,
        totalImagesAvailable: result.total_images_available,
        imagesProcessed: result.images_processed,
        pseudoLabelsAccumulated: result.pseudo_labels_accumulated,
        totalPseudoLabelsGenerated: result.total_pseudo_labels_generated,
        retrainingCount: result.retraining_count,
        currentAccuracy: result.current_accuracy,
        initialAccuracy: result.initial_accuracy,
        pseudoLabelPrecision: result.pseudo_label_precision,
        accuracyHistory: result.accuracy_history.map(([day, acc]: [number, number]) => ({ day, accuracy: acc })),
      }));

      addActivity('success', `Ready! Starting accuracy: ${result.initial_accuracy.toFixed(1)}%`);
      isInitializing = false;
      loadingMessage = '';
    } catch (e) {
      demoState.update(s => ({
        ...s,
        status: 'error',
        errorMessage: String(e),
      }));
      addActivity('error', `Failed to initialize: ${e}`);
      isInitializing = false;
      loadingMessage = '';
    }
  }

  async function advanceDay() {
    if (isProcessing) return;
    
    isProcessing = true;
    loadingMessage = 'Processing today\'s images...';
    demoState.update(s => ({...s, status: 'running'}));

    try {
      await invoke('advance_demo_day');
      // Result comes via event listener
    } catch (e) {
      demoState.update(s => ({
        ...s,
        status: 'error',
        errorMessage: String(e),
      }));
      addActivity('error', `Failed to advance day: ${e}`);
      isProcessing = false;
      loadingMessage = '';
    }
  }

  async function resetSession() {
    await invoke('reset_demo_session');
    demoState.update(s => ({
      status: 'idle',
      currentDay: 0,
      totalImagesAvailable: 0,
      imagesProcessed: 0,
      pseudoLabelsAccumulated: 0,
      totalPseudoLabelsGenerated: 0,
      retrainingCount: 0,
      currentAccuracy: 0,
      initialAccuracy: 0,
      pseudoLabelPrecision: 0,
      accuracyHistory: [],
      lastDayResult: null,
    }));
    addActivity('info', 'Session reset');
  }

  async function processFarmerImages() {
    if (isImportingFarmer || isProcessing) return;
    
    isImportingFarmer = true;
    loadingMessage = 'Processing farmer images...';
    addActivity('info', 'Importing farmer field images...');

    try {
      await invoke('process_farmer_images');
      // Result comes via event listener
    } catch (e) {
      demoState.update(s => ({
        ...s,
        errorMessage: String(e),
      }));
      addActivity('error', `Failed to process farmer images: ${e}`);
      isImportingFarmer = false;
      loadingMessage = '';
    }
  }

  async function manualRetrain() {
    if (isRetraining || isProcessing || isImportingFarmer) return;
    
    isRetraining = true;
    loadingMessage = 'Retraining model with accumulated pseudo-labels...';
    addActivity('info', `Starting manual retraining with ${$demoState.pseudoLabelsAccumulated} pseudo-labels...`);

    try {
      await invoke('manual_retrain_demo');
      // Result comes via event listener
    } catch (e) {
      demoState.update(s => ({
        ...s,
        errorMessage: String(e),
      }));
      addActivity('error', `Failed to retrain: ${e}`);
      isRetraining = false;
      loadingMessage = '';
    }
  }

  let isInitialized = $derived($demoState.status === 'initialized' || $demoState.status === 'running');
  let improvement = $derived($demoState.currentAccuracy - $demoState.initialAccuracy);
  let canRetrain = $derived($demoState.pseudoLabelsAccumulated >= retrainThreshold);

</script>

<div class="p-6 space-y-6">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h2 class="text-2xl font-bold text-gray-800">Interactive SSL Demo</h2>
      <p class="text-gray-500 text-sm mt-1">Day-by-day semi-supervised learning simulation</p>
    </div>
    <div class="flex gap-3">
      {#if !isInitialized}
        <button
          class="btn-secondary flex items-center gap-2"
          onclick={() => showSettings = !showSettings}
        >
          <Settings class="w-4 h-4" />
          Settings
        </button>
        <button
          class="btn-primary flex items-center gap-2"
          onclick={initSession}
          disabled={isInitializing || !modelPath}
        >
          {#if isInitializing}
            <Loader2 class="w-4 h-4 animate-spin" />
            Initializing...
          {:else}
            <Zap class="w-4 h-4" />
            Start Session
          {/if}
        </button>
      {:else}
        <button
          class="btn-secondary flex items-center gap-2"
          onclick={resetSession}
          disabled={isProcessing || isRetraining}
        >
          <RotateCcw class="w-4 h-4" />
          Reset
        </button>
        {#if canRetrain}
          <button
            class="btn-primary flex items-center gap-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500"
            onclick={manualRetrain}
            disabled={isProcessing || isRetraining}
            title="Retrain model with accumulated pseudo-labels"
          >
            {#if isRetraining}
              <Loader2 class="w-5 h-5 animate-spin" />
              Retraining...
            {:else}
              <RefreshCw class="w-5 h-5" />
              <span class="font-semibold">Retrain Now ({$demoState.pseudoLabelsAccumulated} labels)</span>
            {/if}
          </button>
        {/if}
        <button
          class="btn-primary flex items-center gap-2"
          onclick={advanceDay}
          disabled={isProcessing || isRetraining || $demoState.lastDayResult?.remaining_images === 0}
        >
          {#if isProcessing}
            <Loader2 class="w-4 h-4 animate-spin" />
            Processing...
          {:else}
            <Play class="w-4 h-4" />
            Next Day
          {/if}
        </button>
      {/if}
    </div>
  </div>

  <!-- Settings Panel -->
  {#if showSettings && !isInitialized}
    <Card title="Demo Configuration">
      <div class="grid grid-cols-2 gap-4">
        <div class="col-span-2">
          <label class="block text-sm font-medium text-gray-300 mb-2">Base CNN Model (no SSL training yet)</label>
          <div class="flex gap-2">
            <input 
              type="text" 
              class="input flex-1" 
              bind:value={modelPath}
              placeholder="Select a base model..."
              readonly
            />
            <button 
              class="btn-secondary flex items-center gap-2"
              onclick={selectBaseModel}
            >
              <FolderOpen class="w-4 h-4" />
              Browse
            </button>
          </div>
          <p class="text-xs text-gray-500 mt-1">
            Select from output/models/ - these are trained on 20% labeled data only
          </p>
          {#if !modelPath}
            <p class="text-xs text-yellow-400 mt-1">
              Please select a base model to start the demo
            </p>
          {/if}
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Images Per Day</label>
          <input 
            type="number" 
            class="input w-full" 
            bind:value={imagesPerDay}
            min="10"
            max="500"
          />
          <p class="text-xs text-gray-500 mt-1">How many images to process each day</p>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Confidence Threshold</label>
          <input 
            type="number" 
            class="input w-full" 
            bind:value={confidenceThreshold}
            min="0.5"
            max="0.99"
            step="0.05"
          />
          <p class="text-xs text-gray-500 mt-1">Minimum confidence to accept pseudo-labels</p>
        </div>
        <div>
          <label class="block text-sm font-medium text-gray-300 mb-2">Retrain Threshold</label>
          <input 
            type="number" 
            class="input w-full" 
            bind:value={retrainThreshold}
            min="50"
            max="500"
            step="50"
          />
          <p class="text-xs text-gray-500 mt-1">Retrain after accumulating this many pseudo-labels</p>
        </div>
      </div>
    </Card>
  {/if}

  <!-- Loading Indicator -->
  {#if isInitializing || isProcessing || isRetraining}
    <Card>
      <div class="flex items-center gap-4 p-4">
        <Loader2 class="w-8 h-8 text-blue-600 animate-spin" />
        <div>
          <p class="font-semibold text-gray-800">{loadingMessage}</p>
          {#if isRetraining}
            <p class="text-sm text-gray-500">Retraining with 5 epochs - this takes ~6 minutes...</p>
          {:else}
            <p class="text-sm text-gray-500">This may take a moment...</p>
          {/if}
        </div>
      </div>
    </Card>
  {/if}

  <!-- Status Cards -->
  {#if isInitialized && !isInitializing}
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      <Card>
        <div class="flex items-center gap-3 mb-2">
          <Calendar class="w-5 h-5 text-blue-400" />
          <p class="text-gray-400 text-sm">Current Day</p>
        </div>
        <p class="text-3xl font-bold text-white">
          {$demoState.currentDay}
        </p>
      </Card>

      <Card>
        <div class="flex items-center gap-3 mb-2">
          <ImageIcon class="w-5 h-5 text-emerald-400" />
          <p class="text-gray-400 text-sm">Images Processed</p>
        </div>
        <p class="text-3xl font-bold text-emerald-400">
          {$demoState.imagesProcessed}
        </p>
        <p class="text-xs text-gray-500 mt-1">
          {$demoState.lastDayResult?.remaining_images || 0} remaining
        </p>
      </Card>

      <Card>
        <div class="flex items-center gap-3 mb-2">
          <TrendingUp class="w-5 h-5 text-blue-400" />
          <p class="text-gray-400 text-sm">Accuracy</p>
        </div>
        <p class="text-3xl font-bold text-blue-400">
          {$demoState.currentAccuracy.toFixed(1)}%
        </p>
        {#if improvement > 0}
          <p class="text-xs text-green-400 mt-1">
            +{improvement.toFixed(2)}% improvement
          </p>
        {:else if $demoState.initialAccuracy > 0}
          <p class="text-xs text-gray-500 mt-1">
            Started at {$demoState.initialAccuracy.toFixed(1)}%
          </p>
        {/if}
      </Card>

      <Card>
        <div class="flex items-center gap-3 mb-2">
          <Zap class="w-5 h-5 text-yellow-400" />
          <p class="text-gray-400 text-sm">Pseudo-Labels</p>
        </div>
        <p class="text-3xl font-bold text-yellow-400">
          {$demoState.totalPseudoLabelsGenerated}
        </p>
        <div class="mt-2">
          {#if $demoState.pseudoLabelsAccumulated > 0}
            <div class="flex items-center gap-2">
              <div class="flex-1 h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  class="h-full bg-yellow-400 rounded-full transition-all"
                  style="width: {($demoState.pseudoLabelsAccumulated / retrainThreshold * 100)}%"
                ></div>
              </div>
            </div>
            <p class="text-xs text-gray-500 mt-1">
              {$demoState.pseudoLabelsAccumulated} / {retrainThreshold}
              {#if untilRetrain > 0}
                <span class="text-blue-400 font-medium">({untilRetrain} more to retrain)</span>
              {/if}
            </p>
          {:else}
            <p class="text-xs text-gray-500 mt-1">No pseudo-labels queued</p>
          {/if}
        </div>
      </Card>
    </div>

    <!-- Today's Images -->
    {#if $demoState.lastDayResult && $demoState.lastDayResult.sample_images.length > 0}
      {@const hasFarmerImages = $demoState.lastDayResult.sample_images.some(img => img.is_farmer_image)}
      <Card title={hasFarmerImages ? `🌾 Day ${$demoState.currentDay}: Farmer Field Photos (${$demoState.lastDayResult.images_processed_today} images)` : `Day ${$demoState.currentDay}: Stream Images (${$demoState.lastDayResult.images_processed_today} images)`}>
        <div class="grid grid-cols-3 md:grid-cols-5 lg:grid-cols-8 gap-3">
          {#each $demoState.lastDayResult.sample_images as img}
            <div class="relative group">
              <div class="aspect-square bg-gray-100 rounded-lg border-2 {img.accepted ? 'border-green-500' : 'border-gray-300'} {img.is_farmer_image ? 'ring-2 ring-amber-400/50 ring-offset-1 ring-offset-gray-900' : ''} overflow-hidden flex items-center justify-center">
                {#if img.base64_thumbnail}
                  <img 
                    src={img.base64_thumbnail} 
                    alt="Plant leaf" 
                    class="w-full h-full object-cover"
                  />
                {:else}
                  <ImageIcon class="w-6 h-6 text-gray-400" />
                {/if}
              </div>
              <!-- Farmer badge indicator -->
              {#if img.is_farmer_image}
                <div class="absolute -top-1 -left-1 bg-amber-500 rounded-full p-0.5" title="Farmer upload">
                  <Upload class="w-3 h-3 text-white" />
                </div>
              {/if}
              <div class="absolute -top-1 -right-1">
                {#if img.accepted}
                  <CheckCircle2 class="w-5 h-5 text-green-500 bg-white rounded-full" />
                {:else}
                  <XCircle class="w-5 h-5 text-gray-400 bg-white rounded-full" />
                {/if}
              </div>
              <p class="text-xs text-center mt-1 text-gray-600">
                {(img.confidence * 100).toFixed(0)}%
              </p>
            </div>
          {/each}
        </div>
        <div class="mt-4 pt-4 border-t flex items-center justify-between text-sm">
          <div class="flex items-center gap-4">
            <div class="flex items-center gap-2">
              <CheckCircle2 class="w-4 h-4 text-green-500" />
              <span class="text-gray-700 font-medium">
                {$demoState.lastDayResult.pseudo_labels_accepted_today} accepted
              </span>
            </div>
            <div class="flex items-center gap-2">
              <XCircle class="w-4 h-4 text-gray-400" />
              <span class="text-gray-500">
                {$demoState.lastDayResult.images_processed_today - $demoState.lastDayResult.pseudo_labels_accepted_today} rejected
              </span>
            </div>
            {#if hasFarmerImages}
              <div class="flex items-center gap-2">
                <Upload class="w-4 h-4 text-amber-400" />
                <span class="text-amber-400 font-medium">Farmer upload</span>
              </div>
            {/if}
          </div>
          {#if $demoState.lastDayResult.did_retrain}
            <div class="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-xs font-semibold">
              Model Retrained! {$demoState.lastDayResult.accuracy_after_retrain?.toFixed(1)}%
            </div>
          {/if}
        </div>
      </Card>
    {/if}

    <!-- Accuracy Growth Chart -->
    {#if $demoState.accuracyHistory.length > 1}
      <Card title="Model Learning Progress">
        <div class="h-64">
          <LineChart 
            data={accuracyChartData}
            labels={accuracyChartLabels}
            label="Validation Accuracy"
            color="#3b82f6"
            yAxisLabel="Accuracy (%)"
          />
        </div>
      </Card>
    {/if}
  {/if}

  <!-- Getting Started Help -->
  {#if !isInitialized && !isInitializing}
    <Card>
      <div class="text-center py-12">
        <div class="mb-6">
          <div class="inline-block p-4 bg-blue-600/20 rounded-full mb-4">
            <Zap class="w-8 h-8 text-blue-400" />
          </div>
          <h3 class="text-xl font-bold text-white mb-2">Edge SSL Demonstration</h3>
          <p class="text-gray-300 max-w-xl mx-auto">
            Experience how a model learns day-by-day using semi-supervised learning on edge devices. 
            Watch it improve without sending data to the cloud.
          </p>
        </div>
        
        <div class="max-w-2xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 text-left">
          <div class="p-4 bg-background-dark rounded-lg border border-gray-700">
            <div class="font-semibold text-white mb-1">📷 Day 0</div>
            <p class="text-sm text-gray-400">Start with a model trained on 20% labeled data (~89% accuracy)</p>
          </div>
          <div class="p-4 bg-background-dark rounded-lg border border-gray-700">
            <div class="font-semibold text-white mb-1">🌾 Days 1-N</div>
            <p class="text-sm text-gray-400">Farmer uploads field photos daily. High-confidence predictions become pseudo-labels</p>
          </div>
          <div class="p-4 bg-background-dark rounded-lg border border-gray-700">
            <div class="font-semibold text-white mb-1">📈 Retrain</div>
            <p class="text-sm text-gray-400">Model retrains with pseudo-labels every 200 images and accuracy improves</p>
          </div>
        </div>

        <button class="btn-primary btn-lg" onclick={initSession}>
          <Zap class="w-5 h-5" />
          Start Demo
        </button>
      </div>
    </Card>
  {/if}
</div>

<style>
  .btn-lg {
    padding: 12px 32px;
    font-size: 16px;
    display: inline-flex;
    align-items: center;
    gap: 8px;
  }
</style>
