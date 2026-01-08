<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { open } from "@tauri-apps/plugin-dialog";
    import { onMount, onDestroy } from "svelte";
    import LineChart from "$lib/components/LineChart.svelte";

    // Training parameters
    let method = "finetuning";
    let numTasks = 5;
    let epochsPerTask = 20;
    let batchSize = 32;
    let learningRate = 0.001;
    let datasetPath = "";

    // Method-specific parameters
    let ewcLambda = 1000.0;
    let memorySize = 500;
    let distillationTemperature = 2.0;
    let freezeLayers = true;

    // Training state
    let isTraining = false;
    let progress: any = null;
    let progressInterval: any = null;
    let result: any = null;

    // Chart data
    let accuracyHistory: number[] = [];
    let bwtHistory: number[] = [];
    let lossHistory: number[] = [];

    // Available methods
    let methods: any[] = [];

    onMount(async () => {
        // Load available methods
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
            // Start training (runs in background)
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

            // Start polling for progress
            startProgressPolling();

            // Wait for completion
            result = await trainingPromise;

            console.log("Training completed:", result);
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
                const currentProgress = await invoke(
                    "get_incremental_progress",
                );
                if (currentProgress) {
                    progress = currentProgress;

                    // Update charts
                    accuracyHistory = [
                        ...accuracyHistory,
                        progress.taskAccuracy,
                    ];
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

    $: selectedMethod = methods.find((m) => m.id === method);
</script>

<div class="container mx-auto p-6 max-w-7xl">
    <div class="mb-8">
        <h1 class="text-4xl font-bold mb-2">Incremental Learning</h1>
        <p class="text-gray-400">
            Train models incrementally to learn new tasks without forgetting old
            ones
        </p>
    </div>

    <!-- Configuration Card -->
    <div class="card bg-base-300 shadow-xl mb-6">
        <div class="card-body">
            <h2 class="card-title text-2xl mb-4">Configuration</h2>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <!-- Method Selection -->
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold">Method</span>
                    </label>
                    <select
                        class="select select-bordered w-full"
                        bind:value={method}
                        disabled={isTraining}
                    >
                        <option value="finetuning">Fine-Tuning</option>
                        <option value="lwf"
                            >Learning without Forgetting (LwF)</option
                        >
                        <option value="ewc"
                            >Elastic Weight Consolidation (EWC)</option
                        >
                        <option value="rehearsal"
                            >Rehearsal (Memory Replay)</option
                        >
                    </select>
                    {#if selectedMethod}
                        <label class="label">
                            <span class="label-text-alt text-gray-400"
                                >{selectedMethod.description}</span
                            >
                        </label>
                    {/if}
                </div>

                <!-- Dataset Path -->
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold">Dataset</span>
                    </label>
                    <div class="join w-full">
                        <input
                            type="text"
                            placeholder="Select dataset directory..."
                            class="input input-bordered join-item flex-1"
                            bind:value={datasetPath}
                            disabled={isTraining}
                            readonly
                        />
                        <button
                            class="btn join-item"
                            on:click={selectDataset}
                            disabled={isTraining}
                        >
                            Browse
                        </button>
                    </div>
                </div>

                <!-- Number of Tasks -->
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold"
                            >Number of Tasks</span
                        >
                    </label>
                    <input
                        type="number"
                        class="input input-bordered"
                        bind:value={numTasks}
                        min="2"
                        max="10"
                        disabled={isTraining}
                    />
                    <label class="label">
                        <span class="label-text-alt"
                            >Dataset will be split into {numTasks} sequential tasks</span
                        >
                    </label>
                </div>

                <!-- Epochs per Task -->
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold"
                            >Epochs per Task</span
                        >
                    </label>
                    <input
                        type="number"
                        class="input input-bordered"
                        bind:value={epochsPerTask}
                        min="5"
                        max="100"
                        disabled={isTraining}
                    />
                </div>

                <!-- Batch Size -->
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold">Batch Size</span>
                    </label>
                    <input
                        type="number"
                        class="input input-bordered"
                        bind:value={batchSize}
                        min="8"
                        max="128"
                        step="8"
                        disabled={isTraining}
                    />
                </div>

                <!-- Learning Rate -->
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold"
                            >Learning Rate</span
                        >
                    </label>
                    <input
                        type="number"
                        class="input input-bordered"
                        bind:value={learningRate}
                        min="0.0001"
                        max="0.1"
                        step="0.0001"
                        disabled={isTraining}
                    />
                </div>
            </div>

            <!-- Method-specific parameters -->
            {#if method === "ewc"}
                <div class="divider">EWC Parameters</div>
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold"
                            >Lambda (Importance Weight)</span
                        >
                    </label>
                    <input
                        type="number"
                        class="input input-bordered"
                        bind:value={ewcLambda}
                        min="100"
                        max="10000"
                        step="100"
                        disabled={isTraining}
                    />
                    <label class="label">
                        <span class="label-text-alt"
                            >Higher values = more protection of old knowledge</span
                        >
                    </label>
                </div>
            {:else if method === "rehearsal"}
                <div class="divider">Rehearsal Parameters</div>
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold">Memory Size</span
                        >
                    </label>
                    <input
                        type="number"
                        class="input input-bordered"
                        bind:value={memorySize}
                        min="100"
                        max="2000"
                        step="100"
                        disabled={isTraining}
                    />
                    <label class="label">
                        <span class="label-text-alt"
                            >Number of exemplars to store per task</span
                        >
                    </label>
                </div>
            {:else if method === "lwf"}
                <div class="divider">LwF Parameters</div>
                <div class="form-control">
                    <label class="label">
                        <span class="label-text font-semibold"
                            >Distillation Temperature</span
                        >
                    </label>
                    <input
                        type="number"
                        class="input input-bordered"
                        bind:value={distillationTemperature}
                        min="1"
                        max="5"
                        step="0.5"
                        disabled={isTraining}
                    />
                    <label class="label">
                        <span class="label-text-alt"
                            >Higher temperature = softer probability
                            distribution</span
                        >
                    </label>
                </div>
            {:else if method === "finetuning"}
                <div class="divider">Fine-Tuning Parameters</div>
                <div class="form-control">
                    <label class="label cursor-pointer">
                        <span class="label-text">Freeze Early Layers</span>
                        <input
                            type="checkbox"
                            class="toggle"
                            bind:checked={freezeLayers}
                            disabled={isTraining}
                        />
                    </label>
                </div>
            {/if}

            <!-- Action Buttons -->
            <div class="card-actions justify-end mt-6">
                {#if !isTraining}
                    <button
                        class="btn btn-primary btn-lg"
                        on:click={startTraining}
                        disabled={!datasetPath}
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            class="w-6 h-6"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.347a1.125 1.125 0 0 1 0 1.972l-11.54 6.347c-.75.412-1.667-.13-1.667-.986V5.653Z"
                            />
                        </svg>
                        Start Training
                    </button>
                {:else}
                    <button
                        class="btn btn-error btn-lg"
                        on:click={stopTraining}
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke-width="1.5"
                            stroke="currentColor"
                            class="w-6 h-6"
                        >
                            <path
                                stroke-linecap="round"
                                stroke-linejoin="round"
                                d="M5.25 7.5A2.25 2.25 0 0 1 7.5 5.25h9a2.25 2.25 0 0 1 2.25 2.25v9a2.25 2.25 0 0 1-2.25 2.25h-9a2.25 2.25 0 0 1-2.25-2.25v-9Z"
                            />
                        </svg>
                        Stop Training
                    </button>
                {/if}
            </div>
        </div>
    </div>

    <!-- Progress Card -->
    {#if progress}
        <div class="card bg-base-300 shadow-xl mb-6">
            <div class="card-body">
                <h2 class="card-title text-2xl mb-4">Training Progress</h2>

                <!-- Status Message -->
                <div class="alert alert-info mb-4">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        class="stroke-current shrink-0 w-6 h-6"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        ></path>
                    </svg>
                    <span>{progress.status}</span>
                </div>

                <!-- Stats Grid -->
                <div
                    class="stats stats-vertical lg:stats-horizontal shadow w-full mb-6"
                >
                    <div class="stat">
                        <div class="stat-title">Current Task</div>
                        <div class="stat-value text-primary">
                            {progress.currentTask + 1}/{progress.totalTasks}
                        </div>
                        <div class="stat-desc">
                            Epoch {progress.currentEpoch + 1}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Task Accuracy</div>
                        <div class="stat-value">
                            {progress.taskAccuracy.toFixed(2)}%
                        </div>
                        <div class="stat-desc">Current task performance</div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Average Accuracy</div>
                        <div class="stat-value">
                            {progress.averageAccuracy.toFixed(2)}%
                        </div>
                        <div class="stat-desc">Across all tasks</div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Training Loss</div>
                        <div class="stat-value text-sm">
                            {progress.loss.toFixed(4)}
                        </div>
                        <div class="stat-desc">Cross-entropy</div>
                    </div>
                </div>

                <!-- Continual Learning Metrics -->
                <div
                    class="stats stats-vertical lg:stats-horizontal shadow w-full"
                >
                    <div class="stat">
                        <div class="stat-title">Backward Transfer</div>
                        <div
                            class="stat-value text-sm"
                            class:text-success={progress.bwt > 0}
                            class:text-error={progress.bwt < 0}
                        >
                            {progress.bwt.toFixed(3)}
                        </div>
                        <div class="stat-desc">
                            {progress.bwt > 0
                                ? "Positive 👍"
                                : "Negative (forgetting) 👎"}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Forward Transfer</div>
                        <div
                            class="stat-value text-sm"
                            class:text-success={progress.fwt > 0}
                            class:text-error={progress.fwt < 0}
                        >
                            {progress.fwt.toFixed(3)}
                        </div>
                        <div class="stat-desc">
                            {progress.fwt > 0
                                ? "Good generalization 👍"
                                : "Poor transfer 👎"}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Forgetting</div>
                        <div
                            class="stat-value text-sm"
                            class:text-success={progress.forgetting < 0.1}
                            class:text-warning={progress.forgetting >= 0.1 &&
                                progress.forgetting < 0.2}
                            class:text-error={progress.forgetting >= 0.2}
                        >
                            {progress.forgetting.toFixed(3)}
                        </div>
                        <div class="stat-desc">
                            {progress.forgetting < 0.1
                                ? "Low 👍"
                                : progress.forgetting < 0.2
                                  ? "Medium ⚠️"
                                  : "High 👎"}
                        </div>
                    </div>
                </div>

                <!-- Charts -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    <div class="bg-base-100 p-4 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">
                            Accuracy Over Time
                        </h3>
                        <div class="h-64">
                            <LineChart
                                data={accuracyHistory}
                                label="Task Accuracy"
                                color="#10B981"
                            />
                        </div>
                    </div>

                    <div class="bg-base-100 p-4 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">
                            Backward Transfer
                        </h3>
                        <div class="h-64">
                            <LineChart
                                data={bwtHistory}
                                label="Backward Transfer"
                                color="#EF4444"
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    {/if}

    <!-- Results Card -->
    {#if result}
        <div class="card bg-base-300 shadow-xl">
            <div class="card-body">
                <h2 class="card-title text-2xl mb-4">Training Completed! 🎉</h2>

                <div
                    class="stats stats-vertical lg:stats-horizontal shadow w-full mb-4"
                >
                    <div class="stat">
                        <div class="stat-title">Method</div>
                        <div class="stat-value text-sm">
                            {result.method.toUpperCase()}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Final Accuracy</div>
                        <div class="stat-value">
                            {result.finalAccuracy.toFixed(2)}%
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Average Accuracy</div>
                        <div class="stat-value">
                            {result.averageAccuracy.toFixed(2)}%
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Duration</div>
                        <div class="stat-value text-sm">
                            {result.durationSeconds.toFixed(1)}s
                        </div>
                    </div>
                </div>

                <!-- Per-task accuracies -->
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

                <!-- Final metrics summary -->
                <div class="alert alert-success mt-4">
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        class="stroke-current shrink-0 h-6 w-6"
                        fill="none"
                        viewBox="0 0 24 24"
                    >
                        <path
                            stroke-linecap="round"
                            stroke-linejoin="round"
                            stroke-width="2"
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                    </svg>
                    <div>
                        <h3 class="font-bold">Summary</h3>
                        <div class="text-xs">
                            BWT: {result.bwt.toFixed(3)} | FWT: {result.fwt.toFixed(
                                3,
                            )} | Forgetting: {result.forgetting.toFixed(3)} | Intransigence:
                            {result.intransigence.toFixed(3)}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    {/if}
</div>
