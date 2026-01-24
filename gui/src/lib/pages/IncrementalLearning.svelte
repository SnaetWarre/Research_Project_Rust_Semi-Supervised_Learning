<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { open } from "@tauri-apps/plugin-dialog";
    import { onMount, onDestroy } from "svelte";
    import LineChart from "$lib/components/LineChart.svelte";
    import Card from "$lib/components/Card.svelte";
    import { Settings, Play, Square, Folder, CheckCircle, AlertTriangle, TrendingUp, Layers } from "lucide-svelte";

    // Training parameters
    let method = $state("finetuning");
    let numTasks = $state(5);
    let epochsPerTask = $state(20);
    let batchSize = $state(32);
    let learningRate = $state(0.001);
    let datasetPath = $state("");

    // Method-specific parameters
    let ewcLambda = $state(1000.0);
    let memorySize = $state(500);
    let distillationTemperature = $state(2.0);
    let freezeLayers = $state(true);

    // Training state
    let isTraining = $state(false);
    let progress = $state<any>(null);
    let progressInterval: any = null;
    let result = $state<any>(null);

    // Chart data
    let accuracyHistory = $state<number[]>([]);
    let bwtHistory = $state<number[]>([]);
    let lossHistory = $state<number[]>([]);

    // Derived states for charts (fixes Svelte 5 state proxy issue)
    let accuracyHistoryData = $derived([...accuracyHistory]);
    let bwtHistoryData = $derived([...bwtHistory]);
    let lossHistoryData = $derived([...lossHistory]);

    // Available methods
    let methods = $state<any[]>([]);

    onMount(async () => {
        try {
            methods = await invoke("get_incremental_methods");
        } catch (error) {
            console.error("Failed to load methods:", error);
        }
    });

    async function selectDataset() {
        const selected = await open({
            directory: true,
            multiple: false,
            title: "Select PlantVillage Dataset Directory",
        });

        if (selected) {
            datasetPath = selected as string;
        }
    }

    async function startTraining() {
        if (!datasetPath) {
            alert("Please select a dataset directory first");
            return;
        }

        isTraining = true;
        accuracyHistory = [];
        bwtHistory = [];
        lossHistory = [];
        result = null;

        try {
            const trainingPromise = invoke("train_incremental", {
                params: {
                    method,
                    numTasks,
                    epochsPerTask,
                    batchSize,
                    learningRate,
                    datasetPath,
                    ewcLambda: method === "ewc" ? ewcLambda : null,
                    memorySize: method === "rehearsal" ? memorySize : null,
                    distillationTemperature:
                        method === "lwf" ? distillationTemperature : null,
                    freezeLayers: method === "finetuning" ? freezeLayers : null,
                },
            });

            startProgressPolling();
            result = await trainingPromise;
        } catch (error) {
            console.error("Training failed:", error);
            alert(`Training failed: ${error}`);
        } finally {
            isTraining = false;
            stopProgressPolling();
        }
    }

    function startProgressPolling() {
        progressInterval = setInterval(async () => {
            try {
                const currentProgress = await invoke<any>("get_incremental_progress");
                if (currentProgress) {
                    progress = currentProgress;
                    accuracyHistory = [...accuracyHistory, progress.taskAccuracy];
                    bwtHistory = [...bwtHistory, progress.bwt];
                    lossHistory = [...lossHistory, progress.loss];
                }
            } catch (error) {
                console.error("Failed to get progress:", error);
            }
        }, 1000);
    }

    function stopProgressPolling() {
        if (progressInterval) {
            clearInterval(progressInterval);
            progressInterval = null;
        }
    }

    async function stopTraining() {
        try {
            await invoke("stop_incremental_training");
            isTraining = false;
            stopProgressPolling();
        } catch (error) {
            console.error("Failed to stop training:", error);
        }
    }

    onDestroy(() => {
        stopProgressPolling();
    });

    const selectedMethod = $derived(methods.find((m) => m.id === method));
</script>

<div class="p-6 space-y-6">
    <div class="flex items-center justify-between">
        <div>
            <h2 class="text-3xl font-bold text-gray-800">Incremental Learning</h2>
            <p class="text-gray-500 mt-2">Train models sequentially on new tasks</p>
        </div>
    </div>

    <!-- Configuration Card -->
    <Card title="Configuration">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <!-- Method Selection -->
            <div>
                <label class="block text-sm text-gray-500 mb-2">Method</label>
                <select
                    class="input w-full"
                    bind:value={method}
                    disabled={isTraining}
                >
                    <option value="finetuning">Fine-Tuning</option>
                    <option value="lwf">Learning without Forgetting (LwF)</option>
                    <option value="ewc">Elastic Weight Consolidation (EWC)</option>
                    <option value="rehearsal">Rehearsal (Memory Replay)</option>
                </select>
                {#if selectedMethod}
                    <p class="text-xs text-gray-400 mt-1">{selectedMethod.description}</p>
                {/if}
            </div>

            <!-- Dataset Path -->
            <div>
                <label class="block text-sm text-gray-500 mb-2">Dataset</label>
                <div class="flex gap-2">
                    <input
                        type="text"
                        placeholder="Select dataset directory..."
                        class="input flex-1 bg-gray-50"
                        bind:value={datasetPath}
                        disabled={isTraining}
                        readonly
                    />
                    <button
                        class="btn-secondary px-3"
                        onclick={selectDataset}
                        disabled={isTraining}
                    >
                        <Folder class="w-4 h-4" />
                    </button>
                </div>
            </div>

            <!-- Basic Parameters -->
            <div class="grid grid-cols-3 gap-4 md:col-span-2">
                <div>
                    <label class="block text-sm text-gray-500 mb-2">Tasks</label>
                    <input
                        type="number"
                        class="input w-full"
                        bind:value={numTasks}
                        min="2" max="10"
                        disabled={isTraining}
                    />
                </div>
                <div>
                    <label class="block text-sm text-gray-500 mb-2">Epochs/Task</label>
                    <input
                        type="number"
                        class="input w-full"
                        bind:value={epochsPerTask}
                        min="5" max="100"
                        disabled={isTraining}
                    />
                </div>
                <div>
                    <label class="block text-sm text-gray-500 mb-2">Batch Size</label>
                    <input
                        type="number"
                        class="input w-full"
                        bind:value={batchSize}
                        min="8" max="128" step="8"
                        disabled={isTraining}
                    />
                </div>
            </div>

            <!-- Method Specific Params -->
            {#if method === "ewc"}
                <div class="md:col-span-2 border-t pt-4 border-gray-100">
                    <label class="block text-sm font-semibold text-gray-700 mb-3">EWC Settings</label>
                    <div>
                        <label class="block text-sm text-gray-500 mb-2">Lambda (Importance Weight)</label>
                        <input
                            type="number"
                            class="input w-full"
                            bind:value={ewcLambda}
                            min="100" max="10000" step="100"
                            disabled={isTraining}
                        />
                        <p class="text-xs text-gray-400 mt-1">Higher = more protection of old knowledge</p>
                    </div>
                </div>
            {:else if method === "rehearsal"}
                <div class="md:col-span-2 border-t pt-4 border-gray-100">
                    <label class="block text-sm font-semibold text-gray-700 mb-3">Rehearsal Settings</label>
                    <div>
                        <label class="block text-sm text-gray-500 mb-2">Memory Size</label>
                        <input
                            type="number"
                            class="input w-full"
                            bind:value={memorySize}
                            min="100" max="2000" step="100"
                            disabled={isTraining}
                        />
                        <p class="text-xs text-gray-400 mt-1">Total exemplars stored</p>
                    </div>
                </div>
            {:else if method === "lwf"}
                <div class="md:col-span-2 border-t pt-4 border-gray-100">
                    <label class="block text-sm font-semibold text-gray-700 mb-3">LwF Settings</label>
                    <div>
                        <label class="block text-sm text-gray-500 mb-2">Temperature</label>
                        <input
                            type="number"
                            class="input w-full"
                            bind:value={distillationTemperature}
                            min="1" max="5" step="0.5"
                            disabled={isTraining}
                        />
                        <p class="text-xs text-gray-400 mt-1">Softens probability distribution</p>
                    </div>
                </div>
            {/if}

            <div class="md:col-span-2 flex justify-end mt-4">
                {#if !isTraining}
                    <button
                        class="btn-primary flex items-center gap-2"
                        onclick={startTraining}
                        disabled={!datasetPath}
                    >
                        <Play class="w-4 h-4" />
                        Start Training
                    </button>
                {:else}
                    <button
                        class="btn-danger flex items-center gap-2"
                        onclick={stopTraining}
                    >
                        <Square class="w-4 h-4" />
                        Stop Training
                    </button>
                {/if}
            </div>
        </div>
    </Card>

    <!-- Progress Card -->
    {#if progress}
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card>
                <div class="flex items-center gap-3 mb-2">
                    <Layers class="w-5 h-5 text-blue-600" />
                    <p class="text-gray-500 text-sm">Current Task</p>
                </div>
                <p class="text-3xl font-bold text-gray-800">
                    {progress.currentTask + 1}<span class="text-lg text-gray-400">/{progress.totalTasks}</span>
                </p>
                <p class="text-sm text-gray-500 mt-1">Epoch {progress.currentEpoch + 1}</p>
            </Card>

            <Card>
                <div class="flex items-center gap-3 mb-2">
                    <TrendingUp class="w-5 h-5 text-emerald-600" />
                    <p class="text-gray-500 text-sm">Task Accuracy</p>
                </div>
                <p class="text-3xl font-bold text-emerald-600">
                    {progress.taskAccuracy.toFixed(2)}%
                </p>
            </Card>

            <Card>
                <div class="flex items-center gap-3 mb-2">
                    <CheckCircle class="w-5 h-5 text-blue-600" />
                    <p class="text-gray-500 text-sm">Avg Accuracy</p>
                </div>
                <p class="text-3xl font-bold text-blue-600">
                    {progress.averageAccuracy.toFixed(2)}%
                </p>
            </Card>

            <Card>
                <div class="flex items-center gap-3 mb-2">
                    <AlertTriangle class="w-5 h-5 text-yellow-600" />
                    <p class="text-gray-500 text-sm">Forgetting</p>
                </div>
                <p class="text-3xl font-bold {progress.forgetting < 0.1 ? 'text-emerald-600' : 'text-yellow-600'}">
                    {progress.forgetting.toFixed(3)}
                </p>
            </Card>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card title="Accuracy History">
                <div class="h-64">
                     <LineChart
                         data={accuracyHistoryData}
                         label="Task Accuracy"
                         color="#10B981"
                         yAxisLabel="Accuracy (%)"
                     />
                </div>
            </Card>

            <Card title="Backward Transfer">
                <div class="h-64">
                     <LineChart
                         data={bwtHistoryData}
                         label="Backward Transfer"
                         color="#EF4444"
                         yAxisLabel="BWT"
                     />
                </div>
            </Card>
        </div>
    {/if}

    <!-- Final Results -->
    {#if result}
        <Card title="Training Completed! 🎉">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500 mb-1">Final Accuracy</p>
                    <p class="text-2xl font-bold text-gray-800">{result.finalAccuracy.toFixed(2)}%</p>
                </div>
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500 mb-1">Avg Accuracy</p>
                    <p class="text-2xl font-bold text-gray-800">{result.averageAccuracy.toFixed(2)}%</p>
                </div>
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500 mb-1">Backward Transfer</p>
                    <p class="text-2xl font-bold text-gray-800">{result.bwt.toFixed(3)}</p>
                </div>
                <div class="p-4 bg-gray-50 rounded-lg">
                    <p class="text-sm text-gray-500 mb-1">Duration</p>
                    <p class="text-2xl font-bold text-gray-800">{result.durationSeconds.toFixed(1)}s</p>
                </div>
            </div>

            <div class="overflow-x-auto">
                <table class="table">
                    <thead>
                        <tr>
                            <th>Task</th>
                            {#each result.taskAccuracies as _, i}
                                <th>Task {i + 1}</th>
                            {/each}
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="font-semibold">Accuracy</td>
                            {#each result.taskAccuracies as acc}
                                <td>{acc.toFixed(2)}%</td>
                            {/each}
                        </tr>
                    </tbody>
                </table>
            </div>
        </Card>
    {/if}
</div>
