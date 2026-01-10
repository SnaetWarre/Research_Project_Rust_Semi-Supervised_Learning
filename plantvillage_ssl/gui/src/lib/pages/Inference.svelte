<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import Card from '$lib/components/Card.svelte';
  import ImageUpload from '$lib/components/ImageUpload.svelte';
  import ConfidenceBar from '$lib/components/ConfidenceBar.svelte';
  import BarChart from '$lib/components/BarChart.svelte';
  import { modelInfo, addActivity } from '$lib/stores/app';
  import { Clock, CheckCircle, AlertTriangle } from 'lucide-svelte';

  interface PredictionResult {
    predicted_class: number;
    predicted_class_name: string;
    confidence: number;
    probabilities: number[];
    top_5: { class_id: number; class_name: string; probability: number }[];
    inference_time_ms: number;
  }

  let previewUrl = $state<string | null>(null);
  let prediction = $state<PredictionResult | null>(null);
  let isLoading = $state(false);
  let errorMessage = $state<string | null>(null);

  async function handleImageSelected(file: File, dataUrl: string) {
    previewUrl = dataUrl;
    prediction = null;
    errorMessage = null;

    if (!$modelInfo.loaded) {
      errorMessage = 'Please load a model first from the Dashboard';
      return;
    }

    isLoading = true;
    try {
      const base64 = dataUrl.split(',')[1];
      const bytes = atob(base64);
      const arrayBuffer = new Uint8Array(bytes.length);
      for (let i = 0; i < bytes.length; i++) {
        arrayBuffer[i] = bytes.charCodeAt(i);
      }

      prediction = await invoke<PredictionResult>('run_inference_bytes', {
        imageBytes: Array.from(arrayBuffer),
      });

      addActivity('success', `Predicted: ${prediction.predicted_class_name} (${(prediction.confidence * 100).toFixed(1)}%)`);
    } catch (e) {
      errorMessage = String(e);
      addActivity('error', `Inference failed: ${e}`);
    } finally {
      isLoading = false;
    }
  }

  function formatClassName(name: string): string {
    return name.replace(/_/g, ' ').replace(/\s+/g, ' ');
  }

  const confidenceLevel = $derived(() => {
    if (!prediction) return 'unknown';
    if (prediction.confidence >= 0.9) return 'high';
    if (prediction.confidence >= 0.7) return 'medium';
    return 'low';
  });
</script>

<div class="p-6 space-y-6">
  <div class="flex items-center justify-between">
    <h2 class="text-2xl font-bold text-gray-800">Live Inference</h2>
    {#if !$modelInfo.loaded}
      <span class="text-yellow-600 text-sm flex items-center gap-2">
        <AlertTriangle class="w-4 h-4" />
        No model loaded
      </span>
    {/if}
  </div>

  <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
    <!-- Image Upload -->
    <Card title="Input Image">
      <ImageUpload {previewUrl} onImageSelected={handleImageSelected} />
      
      {#if isLoading}
        <div class="mt-4 flex items-center justify-center gap-2 text-blue-600">
          <div class="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
          <span>Processing...</span>
        </div>
      {/if}

      {#if errorMessage}
        <div class="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
          {errorMessage}
        </div>
      {/if}
    </Card>

    <!-- Results -->
    <Card title="Prediction Results">
      {#if prediction}
        <div class="space-y-6">
          <!-- Main Prediction -->
          <div class="text-center">
            <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full {confidenceLevel() === 'high' ? 'bg-emerald-100 text-emerald-700' : confidenceLevel() === 'medium' ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'}">
              <CheckCircle class="w-5 h-5" />
              <span class="font-medium">{formatClassName(prediction.predicted_class_name)}</span>
            </div>
            
            <div class="mt-4">
              <p class="text-5xl font-bold text-gray-800">
                {(prediction.confidence * 100).toFixed(1)}%
              </p>
              <p class="text-gray-500 mt-1">Confidence</p>
            </div>
          </div>

          <!-- Inference Time -->
          <div class="flex items-center justify-center gap-2 text-gray-500">
            <Clock class="w-4 h-4" />
            <span class="text-sm">{prediction.inference_time_ms.toFixed(2)} ms</span>
          </div>

          <!-- Top 5 Predictions -->
          <div>
            <h4 class="text-sm font-medium text-gray-600 mb-3">Top 5 Predictions</h4>
            <div class="space-y-3">
              {#each prediction.top_5 as pred, i}
                <ConfidenceBar 
                  value={pred.probability} 
                  label={`${i + 1}. ${formatClassName(pred.class_name)}`}
                />
              {/each}
            </div>
          </div>
        </div>
      {:else}
        <div class="h-64 flex items-center justify-center text-gray-400">
          <p>Upload an image to see predictions</p>
        </div>
      {/if}
    </Card>
  </div>

  <!-- Probability Distribution -->
  {#if prediction}
    <Card title="Full Class Probability Distribution">
      <div class="h-80">
        <BarChart 
          data={prediction.top_5.map(p => p.probability * 100)}
          labels={prediction.top_5.map(p => formatClassName(p.class_name).substring(0, 20))}
          label="Probability (%)"
          color="#2142f1"
        />
      </div>
    </Card>
  {/if}
</div>
