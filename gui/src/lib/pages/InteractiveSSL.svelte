<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { open } from '@tauri-apps/plugin-dialog';
  import { onMount, onDestroy } from 'svelte';
  import Card from '$lib/components/Card.svelte';
  import LineChart from '$lib/components/LineChart.svelte';
  import { demoState, modelInfo, datasetInfo, addActivity } from '$lib/stores/app';
  import { getBundledModelPath, getBundledDataPath, hasBundledResources, isMobile } from '$lib/bundled';
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

  // Configuration - Optimized for mobile/iPhone
  let dataDir = $datasetInfo?.path || 'data/plantvillage';
  let modelPath = $state('');
  let imagesPerDay = $state(100); // Process 100 images per day
  let confidenceThreshold = $state(0.9);
  let retrainThreshold = $state(150); // Retrain after 150 pseudo-labels
  let labeledRatio = $state(0.2);
  let showSettings = $state(true); // Open by default

  let unlisteners: (() => void)[] = [];
  let isInitializing = $state(false);
  let isProcessing = $state(false);
  let isImportingFarmer = $state(false);
  let isRetraining = $state(false);
  let loadingMessage = $state('');
  let trainingProgress = $state<{
    epoch: number;
    totalEpochs: number;
    batch: number;
    totalBatches: number;
    progressPercent: number;
  } | null>(null);
  
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
    // On mobile/iOS, automatically use bundled resources
    const hasBundled = await hasBundledResources();
    if (hasBundled && isMobile()) {
      try {
        modelPath = (await getBundledModelPath()).replace(/\.mpk$/, '');
        dataDir = await getBundledDataPath();
        addActivity('success', 'Loaded bundled model and dataset for mobile demo');
        showSettings = false; // Hide settings on mobile since they're pre-configured
      } catch (e) {
        console.error('Failed to load bundled resources:', e);
        addActivity('warning', 'Could not load bundled resources');
      }
    }

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
      loadingMessage = `Retraining model with ${data.pseudo_labels} pseudo-labels... (3 epochs)`;
      trainingProgress = null; // Reset progress
      addActivity('info', 'Retraining started - model is learning from pseudo-labels');
    });
    unlisteners.push(retrainingUnlisten);

    // Listen for training progress events
    const trainingProgressUnlisten = await listen('demo:training_progress', (event: any) => {
      const data = event.payload;
      trainingProgress = {
        epoch: data.epoch,
        totalEpochs: data.total_epochs,
        batch: data.batch,
        totalBatches: data.total_batches,
        progressPercent: data.progress_percent,
      };
      console.log('[Training Progress]', trainingProgress);
    });
    unlisteners.push(trainingProgressUnlisten);

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
      trainingProgress = null; // Clear progress
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
      trainingProgress = null; // Clear progress
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
  <!-- Mobile Quick Start Banner -->
  {#if isMobile() && modelPath && !isInitialized}
    <Card>
      <div class="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <h3 class="text-lg font-semibold text-blue-400 mb-2">Mobile Demo Ready!</h3>
        <p class="text-sm text-gray-300 mb-4">
          Pre-loaded model and data. Tap "Start Demo" to begin the SSL simulation.
        </p>
        <button
          class="btn-accent w-full flex items-center justify-center gap-2 h-12 text-lg"
          onclick={initSession}
          disabled={isInitializing}
        >
          {#if isInitializing}
            <Loader2 class="w-5 h-5 animate-spin" />
            Initializing...
          {:else}
            <Play class="w-5 h-5" />
            Start Demo
          {/if}
        </button>
      </div>
    </Card>
  {/if}

  <!-- Header -->
  <div class="flex flex-col gap-4">
    <div>
      <h2 class="text-2xl font-bold text-gray-800">Interactive SSL Demo</h2>
      <p class="text-gray-500 text-sm mt-1">Day-by-day semi-supervised learning simulation</p>
    </div>
    
    {#if !isInitialized && !isMobile()}
      <button
        class="btn-primary flex items-center justify-center gap-2 w-full h-12 text-lg"
        onclick={initSession}
        disabled={isInitializing || !modelPath}
      >
        {#if isInitializing}
          <Loader2 class="w-5 h-5 animate-spin" />
          Initializing...
        {:else}
          <Zap class="w-5 h-5" />
          Start Session
        {/if}
      </button>
    {/if}

    {#if isInitialized}
      <div class="flex flex-col gap-3">
        {#if canRetrain}
          <button
            class="btn-primary flex items-center justify-center gap-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 h-14 text-lg w-full"
            onclick={manualRetrain}
            disabled={isProcessing || isRetraining}
            title="Retrain model with accumulated pseudo-labels"
          >
            {#if isRetraining}
              <Loader2 class="w-6 h-6 animate-spin" />
              Retraining...
            {:else}
              <RefreshCw class="w-6 h-6" />
              <span class="font-semibold">Retrain Now ({$demoState.pseudoLabelsAccumulated} labels)</span>
            {/if}
          </button>
        {/if}
        
        <div class="flex gap-3">
            <button
            class="btn-primary flex-1 flex items-center justify-center gap-2 h-14 text-lg"
            onclick={advanceDay}
            disabled={isProcessing || isRetraining || $demoState.lastDayResult?.remaining_images === 0}
            >
            {#if isProcessing}
                <Loader2 class="w-6 h-6 animate-spin" />
                Processing...
            {:else}
                <Play class="w-6 h-6" />
                Next Day
            {/if}
            </button>
            
            <button
            class="btn-secondary flex items-center justify-center gap-2 h-14 w-14"
            onclick={resetSession}
            disabled={isProcessing || isRetraining}
            >
            <RotateCcw class="w-6 h-6" />
            </button>
        </div>
      </div>
    {/if}
  </div>

  <!-- Loading/Progress Banner -->
  {#if loadingMessage}
    <Card>
      <div class="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <div class="flex items-center gap-3">
          <Loader2 class="w-5 h-5 animate-spin text-blue-400" />
          <div class="flex-1">
            <p class="text-sm font-medium text-blue-400">{loadingMessage}</p>
            {#if trainingProgress}
              <div class="mt-2">
                <div class="flex items-center justify-between text-xs text-gray-400 mb-1">
                  <span>Epoch {trainingProgress.epoch}/{trainingProgress.totalEpochs} - Batch {trainingProgress.batch}/{trainingProgress.totalBatches}</span>
                  <span class="font-semibold text-blue-400">{trainingProgress.progressPercent}%</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                  <div 
                    class="h-full bg-gradient-to-r from-blue-500 to-emerald-500 rounded-full transition-all duration-300"
                    style="width: {trainingProgress.progressPercent}%"
                  ></div>
                </div>
              </div>
            {:else}
              <p class="text-xs text-gray-400 mt-1">Please wait, this may take a moment...</p>
            {/if}
          </div>
        </div>
      </div>
    </Card>
  {/if}

  <!-- Status Cards - Vertical Stack for Mobile -->
  {#if isInitialized && !isInitializing}
    <div class="flex flex-col gap-4">
      <!-- Accuracy Card (Most Important) -->
      <Card>
        <div class="flex items-center justify-between">
            <div>
                <div class="flex items-center gap-2 mb-1">
                <TrendingUp class="w-4 h-4 text-blue-400" />
                <p class="text-gray-400 text-sm">Accuracy</p>
                </div>
                <p class="text-4xl font-bold text-blue-400">
                {$demoState.currentAccuracy.toFixed(1)}%
                </p>
                {#if improvement > 0}
                <p class="text-sm text-green-400 mt-1 font-medium">
                    +{improvement.toFixed(2)}% improvement
                </p>
                {:else if $demoState.initialAccuracy > 0}
                <p class="text-sm text-gray-500 mt-1">
                    Started at {$demoState.initialAccuracy.toFixed(1)}%
                </p>
                {/if}
            </div>
            
            <div class="text-right">
                <div class="flex items-center justify-end gap-2 mb-1">
                    <Calendar class="w-4 h-4 text-gray-400" />
                    <p class="text-gray-400 text-sm">Day</p>
                </div>
                <p class="text-4xl font-bold text-white">{$demoState.currentDay}</p>
            </div>
        </div>
      </Card>

      <!-- Stats Grid -->
      <div class="grid grid-cols-2 gap-4">
        <Card>
            <div class="flex flex-col items-center text-center py-2">
            <ImageIcon class="w-6 h-6 text-emerald-400 mb-2" />
            <p class="text-xs text-gray-400 uppercase tracking-wide">Processed</p>
            <p class="text-2xl font-bold text-white mt-1">
                {$demoState.imagesProcessed}
            </p>
            </div>
        </Card>

        <Card>
            <div class="flex flex-col items-center text-center py-2">
            <Zap class="w-6 h-6 text-yellow-400 mb-2" />
            <p class="text-xs text-gray-400 uppercase tracking-wide">Pseudo-Labels</p>
            <p class="text-2xl font-bold text-white mt-1">
                {$demoState.totalPseudoLabelsGenerated}
            </p>
            </div>
        </Card>
      </div>

      <!-- Retrain Progress -->
      <Card>
        <div class="flex items-center justify-between mb-2">
            <p class="text-sm font-medium text-gray-300">Retrain Progress</p>
            <p class="text-xs text-gray-500">{$demoState.pseudoLabelsAccumulated} / {retrainThreshold}</p>
        </div>
        <div class="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
            <div 
                class="h-full bg-gradient-to-r from-yellow-600 to-yellow-400 rounded-full transition-all"
                style="width: {Math.min(100, ($demoState.pseudoLabelsAccumulated / retrainThreshold * 100))}%"
            ></div>
        </div>
        {#if untilRetrain > 0}
            <p class="text-xs text-center text-gray-500 mt-2">
                Need {untilRetrain} more labels to retrain
            </p>
        {:else}
             <p class="text-xs text-center text-green-400 mt-2 font-medium">
                Ready to retrain!
            </p>
        {/if}
      </Card>
    </div>

    <!-- Today's Images -->
    {#if $demoState.lastDayResult && $demoState.lastDayResult.sample_images.length > 0}
      {@const hasFarmerImages = $demoState.lastDayResult.sample_images.some(img => img.is_farmer_image)}
      <Card title={hasFarmerImages ? `Day ${$demoState.currentDay}: Farmer Photos` : `Day ${$demoState.currentDay}: Stream Images`}>
        <div class="grid grid-cols-4 gap-2">
          {#each $demoState.lastDayResult.sample_images.slice(0, 12) as img}
            <div class="relative group aspect-square">
              <div class="w-full h-full bg-gray-800 rounded-lg overflow-hidden border-2 {img.accepted ? 'border-green-500' : 'border-gray-700'} relative">
                {#if img.base64_thumbnail}
                  <img 
                    src={img.base64_thumbnail} 
                    alt="Plant leaf" 
                    class="w-full h-full object-cover"
                  />
                {:else}
                  <div class="w-full h-full flex items-center justify-center">
                    <ImageIcon class="w-6 h-6 text-gray-600" />
                  </div>
                {/if}
                
                <div class="absolute bottom-0 left-0 right-0 bg-black/60 backdrop-blur-sm py-1 px-1">
                    <p class="text-[10px] text-center text-white font-medium">
                        {(img.confidence * 100).toFixed(0)}%
                    </p>
                </div>
              </div>
              
              <div class="absolute -top-1 -right-1 z-10">
                {#if img.accepted}
                  <CheckCircle2 class="w-4 h-4 text-green-500 bg-white rounded-full" />
                {/if}
              </div>
            </div>
          {/each}
        </div>
        
        <div class="mt-4 pt-4 border-t border-gray-800 flex items-center justify-between text-xs">
            <span class="text-green-400 font-medium">
                {$demoState.lastDayResult.pseudo_labels_accepted_today} accepted
            </span>
            <span class="text-gray-500">
                {$demoState.lastDayResult.images_processed_today - $demoState.lastDayResult.pseudo_labels_accepted_today} rejected
            </span>
        </div>
      </Card>
    {/if}

    <!-- Accuracy Growth Chart -->
    {#if $demoState.accuracyHistory.length > 1}
      <Card title="Model Progress">
        <div class="h-48">
          <LineChart 
            data={accuracyChartData}
            labels={accuracyChartLabels}
            label="Accuracy"
            color="#3b82f6"
            yAxisLabel="%"
          />
        </div>
      </Card>
    {/if}
  {/if}
</div>

<style>
  /* Mobile-specific styles */
  :global(.card) {
    border-radius: 16px !important;
  }
  
  :global(.btn-primary) {
    border-radius: 12px !important;
  }
  
  :global(.btn-secondary) {
    border-radius: 12px !important;
  }
</style>
