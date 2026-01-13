<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import Card from '$lib/components/Card.svelte';
  import BarChart from '$lib/components/BarChart.svelte';
  import { modelInfo, datasetInfo, addActivity, diagnosticsState, type DiagnosticResult } from '$lib/stores/app';
  import { AlertTriangle, CheckCircle, TrendingUp, AlertCircle, RefreshCw, FileImage, Clock } from 'lucide-svelte';

  let isRunning = $state(false);
  
  // Use store values for config
  let numSamples = $state($diagnosticsState.config.numSamples);
  let confidenceThreshold = $state($diagnosticsState.config.confidenceThreshold);
  
  // Get diagnostics from store
  const diagnostics = $derived($diagnosticsState.result);
  const lastRunAt = $derived($diagnosticsState.lastRunAt);

  async function runDiagnostics() {
    if (!$modelInfo.loaded) {
      addActivity('error', 'Please load a model first');
      return;
    }

    if (!$datasetInfo) {
      addActivity('error', 'Please load a dataset first');
      return;
    }

    isRunning = true;
    try {
      const result = await invoke<DiagnosticResult>('run_model_diagnostics', {
        dataDir: $datasetInfo.path,
        numSamples,
        confidenceThreshold,
      });

      // Store results in global state
      diagnosticsState.update(state => ({
        ...state,
        result,
        lastRunAt: new Date(),
        config: { numSamples, confidenceThreshold },
      }));
      
      addActivity('success', `Diagnostics complete: analyzed ${result.total_predictions} predictions`);
    } catch (e) {
      addActivity('error', `Diagnostics failed: ${e}`);
    } finally {
      isRunning = false;
    }
  }

  const biasLevel = $derived.by(() => {
    if (!diagnostics) return 'unknown';
    if (diagnostics.prediction_bias_score > 0.5) return 'high';
    if (diagnostics.prediction_bias_score > 0.3) return 'medium';
    return 'low';
  });

  const biasColor = $derived.by(() => {
    const level = biasLevel;
    if (level === 'high') return 'text-red-500';
    if (level === 'medium') return 'text-yellow-500';
    return 'text-emerald-500';
  });

  const classPredictionData = $derived.by(() => {
    if (!diagnostics) return { labels: [], data: [] };

    const entries = Object.entries(diagnostics.class_predictions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15);

    return {
      labels: entries.map(([classId]) => getClassName(parseInt(classId))),
      data: entries.map(([, count]) => count),
    };
  });

  const confidenceData = $derived.by(() => {
    if (!diagnostics) return { labels: [], data: [] };

    const avgConfidences = Object.entries(diagnostics.class_confidences)
      .map(([classId, confidences]) => {
        const avg = confidences.reduce((sum, c) => sum + c, 0) / confidences.length;
        return [parseInt(classId), avg] as [number, number];
      })
      .sort((a, b) => a[1] - b[1])
      .slice(0, 10);

    return {
      labels: avgConfidences.map(([classId]) => getClassName(classId)),
      data: avgConfidences.map(([, conf]) => conf * 100),
    };
  });

  const inputDistributionData = $derived.by(() => {
    if (!diagnostics) return { labels: [], data: [] };

    const entries = Object.entries(diagnostics.input_distribution)
      .sort((a, b) => b[1] - a[1]);

    return {
      labels: entries.map(([className]) => className.replace(/_/g, ' ')),
      data: entries.map(([, count]) => count),
    };
  });

  function getClassName(classId: number): string {
    const names = [
      "Apple___Apple_scab",
      "Apple___Black_rot",
      "Apple___Cedar_apple_rust",
      "Apple___healthy",
      "Blueberry___healthy",
      "Cherry_(including_sour)___healthy",
      "Cherry_(including_sour)___Powdery_mildew",
      "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
      "Corn_(maize)___Common_rust_",
      "Corn_(maize)___healthy",
      "Corn_(maize)___Northern_Leaf_Blight",
      "Grape___Black_rot",
      "Grape___Esca_(Black_Measles)",
      "Grape___healthy",
      "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
      "Orange___Haunglongbing_(Citrus_greening)",
      "Peach___Bacterial_spot",
      "Peach___healthy",
      "Pepper,_bell___Bacterial_spot",
      "Pepper,_bell___healthy",
      "Potato___Early_blight",
      "Potato___healthy",
      "Potato___Late_blight",
      "Raspberry___healthy",
      "Soybean___healthy",
      "Squash___Powdery_mildew",
      "Strawberry___healthy",
      "Strawberry___Leaf_scorch",
      "Tomato___Bacterial_spot",
      "Tomato___Early_blight",
      "Tomato___healthy",
      "Tomato___Late_blight",
      "Tomato___Leaf_Mold",
      "Tomato___Septoria_leaf_spot",
      "Tomato___Spider_mites Two-spotted_spider_mite",
      "Tomato___Target_Spot",
      "Tomato___Tomato_mosaic_virus",
      "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ];

    return names[classId]?.replace(/_/g, ' ') || `Class ${classId}`;
  }

  function formatClassName(name: string): string {
    return name.substring(0, 30) + (name.length > 30 ? '...' : '');
  }
</script>

<div class="p-6 space-y-6 page-transition">
  <div class="flex items-center justify-between">
    <div>
      <h2 class="text-3xl font-bold" style="color: var(--text-main)">Model Diagnostics</h2>
      <p class="mt-2" style="color: var(--text-secondary)">Analyze model behavior and detect prediction bias</p>
      {#if lastRunAt}
        <p class="text-sm mt-1 flex items-center gap-1" style="color: var(--text-secondary)">
          <Clock class="w-3 h-3" />
          Last run: {lastRunAt.toLocaleString()}
        </p>
      {/if}
    </div>
    <div class="flex gap-3">
      <button
        class="btn-primary flex items-center gap-2"
        onclick={runDiagnostics}
        disabled={isRunning || !$modelInfo.loaded || !$datasetInfo}
      >
        <RefreshCw class="w-4 h-4 {isRunning ? 'animate-spin' : ''}" />
        {isRunning ? 'Running...' : 'Run Diagnostics'}
      </button>
    </div>
  </div>

  <!-- Configuration -->
  <Card>
    <div class="flex items-center gap-2 mb-4">
      <AlertCircle class="w-5 h-5" style="color: var(--c-accent)" />
      <h3 class="text-lg font-semibold" style="color: var(--text-main)">Configuration</h3>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <label class="block text-sm mb-2" style="color: var(--text-secondary)">Number of Samples to Test</label>
        <input
          type="number"
          class="input w-full"
          bind:value={numSamples}
          min="10"
          max="1000"
          step="10"
        />
        <p class="text-xs mt-1" style="color: var(--text-secondary)">More samples = more accurate results (slower)</p>
      </div>
      <div>
        <label class="block text-sm mb-2" style="color: var(--text-secondary)">Confidence Threshold</label>
        <input
          type="number"
          class="input w-full"
          bind:value={confidenceThreshold}
          min="0.1"
          max="1.0"
          step="0.05"
        />
        <p class="text-xs mt-1" style="color: var(--text-secondary)">Predictions below this are considered low confidence</p>
      </div>
    </div>
  </Card>

  {#if diagnostics}
    <!-- Overview Stats -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 animate-fade-in">
      <Card>
        <div class="flex items-start justify-between">
          <div>
            <p class="text-sm" style="color: var(--text-secondary)">Total Predictions</p>
            <p class="text-4xl font-bold mt-2" style="color: var(--text-main)">
              {diagnostics.total_predictions}
            </p>
            <p class="text-sm mt-1" style="color: var(--text-secondary)">samples analyzed</p>
          </div>
          <div class="p-2 rounded-full" style="background: var(--c-accent-dim)">
            <TrendingUp class="w-6 h-6" style="color: var(--c-accent)" />
          </div>
        </div>
      </Card>

      <Card>
        <div class="flex items-start justify-between">
          <div>
            <p class="text-sm" style="color: var(--text-secondary)">Most Predicted</p>
            <p class="text-2xl font-bold mt-2" style="color: var(--text-main)">
              {formatClassName(diagnostics.most_predicted_class_name)}
            </p>
            <p class="text-sm mt-1" style="color: var(--text-secondary)">
              {diagnostics.class_predictions[diagnostics.most_predicted_class]} times
            </p>
          </div>
          <div class="p-2 rounded-full badge-warning">
            <AlertTriangle class="w-6 h-6" />
          </div>
        </div>
      </Card>

      <Card>
        <div class="flex items-start justify-between">
          <div>
            <p class="text-sm" style="color: var(--text-secondary)">Prediction Bias</p>
            <p class="text-4xl font-bold {biasColor} mt-2">
              {(diagnostics.prediction_bias_score * 100).toFixed(1)}%
            </p>
            <p class="text-sm mt-1 capitalize" style="color: var(--text-secondary)">
              {biasLevel} bias detected
            </p>
          </div>
          <div class="p-2 rounded-full {biasLevel === 'high' ? 'badge-error' : biasLevel === 'medium' ? 'badge-warning' : 'badge-success'}">
            {#if biasLevel === 'high'}
              <AlertTriangle class="w-6 h-6" />
            {:else if biasLevel === 'medium'}
              <AlertCircle class="w-6 h-6" />
            {:else}
              <CheckCircle class="w-6 h-6" />
            {/if}
          </div>
        </div>
      </Card>

      <Card>
        <div class="flex items-start justify-between">
          <div>
            <p class="text-sm" style="color: var(--text-secondary)">Low Confidence</p>
            <p class="text-4xl font-bold mt-2" style="color: var(--text-main)">
              {diagnostics.low_confidence_count}
            </p>
            <p class="text-sm mt-1" style="color: var(--text-secondary)">
              {((diagnostics.low_confidence_count / diagnostics.total_predictions) * 100).toFixed(1)}% of total
            </p>
          </div>
          <div class="p-2 rounded-full badge-warning">
            <AlertCircle class="w-6 h-6" />
          </div>
        </div>
      </Card>
    </div>

    <!-- Bias Warning -->
    {#if biasLevel === 'high'}
      <div class="alert badge-error animate-fade-in border-none p-4">
        <div class="flex items-start gap-3">
          <AlertTriangle class="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <h4 class="font-semibold mb-1">High Prediction Bias Detected!</h4>
            <p class="text-sm">
              Your model is strongly biased towards <strong>{diagnostics.most_predicted_class_name}</strong>.
              This suggests:
            </p>
            <ul class="list-disc list-inside text-sm mt-2 space-y-1">
              <li>Training data may be imbalanced (too many samples of this class)</li>
              <li>Model architecture might be too weak to learn discriminative features</li>
              <li>Consider retraining with class weights or balanced dataset</li>
              <li>Try increasing model capacity (base_filters, FC layers)</li>
            </ul>
          </div>
        </div>
      </div>
    {:else if biasLevel === 'medium'}
      <div class="alert badge-warning animate-fade-in border-none p-4">
        <div class="flex items-start gap-3">
          <AlertCircle class="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <h4 class="font-semibold mb-1">Moderate Prediction Bias</h4>
            <p class="text-sm">
              The model shows some preference for certain classes. Monitor this during training and consider
              using class-weighted loss or data augmentation.
            </p>
          </div>
        </div>
      </div>
    {:else}
      <div class="alert badge-success animate-fade-in border-none p-4">
        <div class="flex items-start gap-3">
          <CheckCircle class="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <h4 class="font-semibold mb-1">Low Bias - Model Looks Healthy</h4>
            <p class="text-sm">
              The model's predictions are well-distributed across classes. This indicates good training balance.
            </p>
          </div>
        </div>
      </div>
    {/if}

    <!-- Charts -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <h3 class="text-lg font-semibold mb-4 flex items-center gap-2" style="color: var(--text-main)">
          <TrendingUp class="w-5 h-5" style="color: var(--c-accent)" />
          Top 15 Predicted Classes
        </h3>
        <div class="h-96">
          <BarChart
            data={classPredictionData.data}
            labels={classPredictionData.labels.map(l => formatClassName(l))}
            label="Prediction Count"
            color="#3b82f6"
          />
        </div>
        <p class="text-xs mt-3" style="color: var(--text-secondary)">
          Shows which classes the model predicts most often. Ideally should match dataset distribution.
        </p>
      </Card>

      <Card>
        <h3 class="text-lg font-semibold mb-4 flex items-center gap-2" style="color: var(--text-main)">
          <AlertCircle class="w-5 h-5 text-yellow-500" />
          Lowest Confidence Classes
        </h3>
        <div class="h-96">
          <BarChart
            data={confidenceData.data}
            labels={confidenceData.labels.map(l => formatClassName(l))}
            label="Average Confidence (%)"
            color="#f59e0b"
          />
        </div>
        <p class="text-xs mt-3" style="color: var(--text-secondary)">
          Classes with lowest average confidence. These may need more training data or better features.
        </p>
      </Card>
    </div>

    <!-- Input Distribution -->
    <Card>
      <h3 class="text-lg font-semibold mb-4 flex items-center gap-2" style="color: var(--text-main)">
        <FileImage class="w-5 h-5 text-purple-600" />
        Input Sample Distribution
        <span class="text-sm font-normal" style="color: var(--text-secondary)">
          ({Object.keys(diagnostics.input_distribution).length} classes sampled)
        </span>
      </h3>
      <p class="text-sm mb-4" style="color: var(--text-secondary)">
        Shows which images were randomly selected for diagnostics. Use this to verify if bias comes from input sampling or model predictions.
      </p>
      
      {#if inputDistributionData.labels.length > 0}
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Chart -->
          <div class="h-96">
            <BarChart
              data={inputDistributionData.data}
              labels={inputDistributionData.labels.map(l => formatClassName(l))}
              label="Sample Count"
              color="#8b5cf6"
            />
          </div>
          
          <!-- Table view for detailed counts -->
          <div class="overflow-auto max-h-96 border rounded-xl" style="border-color: var(--border-base)">
            <table class="w-full text-sm">
              <thead class="sticky top-0" style="background: var(--bg-surface)">
                <tr>
                  <th class="text-left px-4 py-3 font-semibold" style="color: var(--text-main)">Class Name</th>
                  <th class="text-right px-4 py-3 font-semibold" style="color: var(--text-main)">Count</th>
                  <th class="text-right px-4 py-3 font-semibold" style="color: var(--text-main)">%</th>
                </tr>
              </thead>
              <tbody class="divide-y" style="divide-color: var(--border-base)">
                {#each inputDistributionData.labels as label, i}
                  <tr class="transition-colors hover:bg-[var(--bg-hover)]">
                    <td class="px-4 py-2" style="color: var(--text-main)">{label}</td>
                    <td class="px-4 py-2 text-right font-mono" style="color: var(--text-secondary)">{inputDistributionData.data[i]}</td>
                    <td class="px-4 py-2 text-right font-mono" style="color: var(--text-secondary)">
                      {((inputDistributionData.data[i] / diagnostics.total_predictions) * 100).toFixed(1)}%
                    </td>
                  </tr>
                {/each}
              </tbody>
            </table>
          </div>
        </div>
      {:else}
        <p class="text-gray-500">No input distribution data available.</p>
      {/if}
    </Card>

    <!-- Recommendations -->
    <Card>
      <h3 class="text-lg font-semibold mb-4 flex items-center gap-2" style="color: var(--text-main)">
        <CheckCircle class="w-5 h-5" style="color: var(--c-accent)" />
        Recommendations
      </h3>
      <div class="space-y-4">
        {#if biasLevel === 'high'}
          <div class="p-4 rounded-xl badge-error">
            <h4 class="font-semibold mb-2">🚨 Critical: Model Retraining Needed</h4>
            <ul class="space-y-2 text-sm">
              <li>• Use the <strong>balanced dataset</strong> in <code class="text-blue-600">data/plantvillage/balanced</code></li>
              <li>• Enable <strong>class-weighted loss</strong> in training settings</li>
              <li>• Increase model capacity: <strong>base_filters=32</strong>, <strong>dropout=0.3</strong></li>
              <li>• Train for <strong>50-100 epochs</strong> with early stopping</li>
              <li>• Use <strong>data augmentation</strong> (rotation, flip, color jitter)</li>
            </ul>
          </div>
        {/if}

        <div class="p-4 rounded-xl" style="background: var(--c-accent-dim); border: 1px solid var(--c-accent-dim);">
          <h4 class="font-semibold mb-2" style="color: var(--c-accent)">💡 Model Architecture Tips</h4>
          <ul class="space-y-2 text-sm" style="color: var(--text-main)">
            <li>• Current recommended config: base_filters=32, dropout=0.3, FC layer=256 units</li>
            <li>• For better performance: Consider base_filters=64 or deeper architecture</li>
            <li>• Monitor validation accuracy - should be 85%+ for good model</li>
          </ul>
        </div>

        <div class="p-4 rounded-xl" style="background: var(--bg-surface); border: 1px solid var(--border-base)">
          <h4 class="font-semibold mb-2" style="color: var(--text-main)">✅ Best Practices</h4>
          <ul class="space-y-2 text-sm" style="color: var(--text-secondary)">
            <li>• Always validate on held-out test set</li>
            <li>• Run diagnostics after every training session</li>
            <li>• Monitor both training loss AND validation accuracy</li>
            <li>• Use learning rate scheduling (cosine annealing)</li>
            <li>• Save checkpoints regularly</li>
          </ul>
        </div>
      </div>
    </Card>
  {:else}
    <Card>
      <div class="h-96 flex flex-col items-center justify-center text-center">
        <div class="w-20 h-20 rounded-full flex items-center justify-center mb-4" style="background: var(--c-accent-dim)">
          <AlertCircle class="w-10 h-10" style="color: var(--c-accent)" />
        </div>
        <h3 class="text-xl font-semibold mb-2" style="color: var(--text-main)">No Diagnostics Available</h3>
        <p class="mb-6 max-w-md" style="color: var(--text-secondary)">
          Load a model and dataset, then run diagnostics to analyze your model's behavior
          and detect any prediction biases.
        </p>
        {#if !$modelInfo.loaded}
          <p class="text-sm" style="color: var(--warning)">⚠️ Please load a model first</p>
        {/if}
        {#if !$datasetInfo}
          <p class="text-sm" style="color: var(--warning)">⚠️ Please load a dataset first</p>
        {/if}
      </div>
    </Card>
  {/if}
</div>
