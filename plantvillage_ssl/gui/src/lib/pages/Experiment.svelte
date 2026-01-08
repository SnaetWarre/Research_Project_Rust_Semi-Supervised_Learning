<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { open } from "@tauri-apps/plugin-dialog";
    import BarChart from "$lib/components/BarChart.svelte";

    // Experiment parameters
    let selectedMethods = ["finetuning", "lwf", "ewc", "rehearsal"];
    let numTasks = 5;
    let epochsPerTask = 20;
    let datasetPath = "";

    // Experiment state
    let isRunning = false;
    let results: any[] = [];
    let currentMethod = "";

    const allMethods = [
        { id: "finetuning", name: "Fine-Tuning", color: "#ef4444" },
        { id: "lwf", name: "Learning without Forgetting", color: "#f59e0b" },
        { id: "ewc", name: "Elastic Weight Consolidation", color: "#10b981" },
        { id: "rehearsal", name: "Rehearsal", color: "#3b82f6" },
    ];

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

    async function runExperiment() {
        if (!datasetPath) {
            alert("Please select a dataset directory first");
            return;
        }

        if (selectedMethods.length === 0) {
            alert("Please select at least one method to compare");
            return;
        }

        isRunning = true;
        results = [];
        currentMethod = "";

        try {
            // Run experiment (this will train all methods sequentially)
            const experimentResults = await invoke("run_experiment", {
                params: {
                    methods: selectedMethods,
                    numTasks,
                    epochsPerTask,
                    datasetPath,
                },
            });

            results = experimentResults as any[];
            console.log("Experiment completed:", results);
        } catch (error) {
            console.error("Experiment failed:", error);
            alert(`Experiment failed: ${error}`);
        } finally {
            isRunning = false;
            currentMethod = "";
        }
    }

    function getBestMethod(metric: string): string {
        if (results.length === 0) return "-";

        let best = results[0];
        for (const result of results) {
            if (
                metric === "accuracy" &&
                result.finalAccuracy > best.finalAccuracy
            ) {
                best = result;
            } else if (metric === "bwt" && result.bwt > best.bwt) {
                best = result;
            } else if (
                metric === "forgetting" &&
                result.forgetting < best.forgetting
            ) {
                best = result;
            }
        }

        return best.method.toUpperCase();
    }
</script>

<div class="container mx-auto p-6 max-w-7xl">
    <div class="mb-8">
        <h1 class="text-4xl font-bold mb-2">Method Comparison Experiment</h1>
        <p class="text-gray-400">
            Compare different incremental learning methods on the same dataset
            and tasks
        </p>
    </div>

    <!-- Configuration Card -->
    <div class="card bg-base-300 shadow-xl mb-6">
        <div class="card-body">
            <h2 class="card-title text-2xl mb-4">Experiment Configuration</h2>

            <!-- Dataset Selection -->
            <div class="form-control mb-4">
                <label class="label">
                    <span class="label-text font-semibold">Dataset</span>
                </label>
                <div class="join w-full">
                    <input
                        type="text"
                        placeholder="Select dataset directory..."
                        class="input input-bordered join-item flex-1"
                        bind:value={datasetPath}
                        disabled={isRunning}
                        readonly
                    />
                    <button
                        class="btn join-item"
                        on:click={selectDataset}
                        disabled={isRunning}
                    >
                        Browse
                    </button>
                </div>
            </div>

            <!-- Method Selection -->
            <div class="form-control mb-4">
                <label class="label">
                    <span class="label-text font-semibold"
                        >Methods to Compare</span
                    >
                </label>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                    {#each allMethods as method}
                        <label class="label cursor-pointer justify-start gap-3">
                            <input
                                type="checkbox"
                                class="checkbox"
                                bind:group={selectedMethods}
                                value={method.id}
                                disabled={isRunning}
                            />
                            <span class="label-text">{method.name}</span>
                        </label>
                    {/each}
                </div>
            </div>

            <!-- Task Configuration -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                        disabled={isRunning}
                    />
                </div>

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
                        disabled={isRunning}
                    />
                </div>
            </div>

            <!-- Action Buttons -->
            <div class="card-actions justify-end mt-6">
                {#if !isRunning}
                    <button
                        class="btn btn-primary btn-lg"
                        on:click={runExperiment}
                        disabled={!datasetPath || selectedMethods.length === 0}
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
                                d="M9.75 3.104v5.714a2.25 2.25 0 0 1-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 0 1 4.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0 1 12 16a9.065 9.065 0 0 1-6.23-.693L5 14.5m14.8.8 1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0 1 12 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5"
                            />
                        </svg>
                        Run Experiment
                    </button>
                {:else}
                    <button class="btn btn-disabled btn-lg" disabled>
                        <span class="loading loading-spinner"></span>
                        Running Experiment...
                    </button>
                {/if}
            </div>
        </div>
    </div>

    <!-- Running Status -->
    {#if isRunning}
        <div class="alert alert-info mb-6">
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
            <span
                >Training {selectedMethods.length} methods sequentially. This may
                take several minutes...</span
            >
        </div>
    {/if}

    <!-- Results -->
    {#if results.length > 0}
        <!-- Summary Stats -->
        <div class="card bg-base-300 shadow-xl mb-6">
            <div class="card-body">
                <h2 class="card-title text-2xl mb-4">Results Summary</h2>

                <div
                    class="stats stats-vertical lg:stats-horizontal shadow w-full mb-4"
                >
                    <div class="stat">
                        <div class="stat-title">Best Accuracy</div>
                        <div class="stat-value text-primary">
                            {getBestMethod("accuracy")}
                        </div>
                        <div class="stat-desc">
                            {Math.max(
                                ...results.map((r) => r.finalAccuracy),
                            ).toFixed(2)}%
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Best BWT</div>
                        <div class="stat-value text-success">
                            {getBestMethod("bwt")}
                        </div>
                        <div class="stat-desc">
                            {Math.max(...results.map((r) => r.bwt)).toFixed(3)} (least
                            forgetting)
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Lowest Forgetting</div>
                        <div class="stat-value text-success">
                            {getBestMethod("forgetting")}
                        </div>
                        <div class="stat-desc">
                            {Math.min(
                                ...results.map((r) => r.forgetting),
                            ).toFixed(3)}
                        </div>
                    </div>

                    <div class="stat">
                        <div class="stat-title">Methods Tested</div>
                        <div class="stat-value">{results.length}</div>
                        <div class="stat-desc">
                            {numTasks} tasks × {epochsPerTask} epochs
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Comparison Charts -->
        <div class="card bg-base-300 shadow-xl mb-6">
            <div class="card-body">
                <h2 class="card-title text-2xl mb-4">Visual Comparison</h2>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-base-100 p-4 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">
                            Final Accuracy
                        </h3>
                        <div class="h-64">
                            <BarChart
                                data={results.map((r) => r.finalAccuracy)}
                                labels={results.map((r) =>
                                    r.method.toUpperCase(),
                                )}
                                label="Final Accuracy (%)"
                                color="#10B981"
                            />
                        </div>
                    </div>

                    <div class="bg-base-100 p-4 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">
                            Average Accuracy
                        </h3>
                        <div class="h-64">
                            <BarChart
                                data={results.map((r) => r.averageAccuracy)}
                                labels={results.map((r) =>
                                    r.method.toUpperCase(),
                                )}
                                label="Average Accuracy (%)"
                                color="#10B981"
                            />
                        </div>
                    </div>

                    <div class="bg-base-100 p-4 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">
                            Backward Transfer
                        </h3>
                        <div class="h-64">
                            <BarChart
                                data={results.map((r) => r.bwt)}
                                labels={results.map((r) =>
                                    r.method.toUpperCase(),
                                )}
                                label="Backward Transfer"
                                color="#3B82F6"
                            />
                        </div>
                        <p class="text-xs text-gray-400 mt-2">
                            Positive = good, Negative = forgetting
                        </p>
                    </div>

                    <div class="bg-base-100 p-4 rounded-lg">
                        <h3 class="text-lg font-semibold mb-2">Forgetting</h3>
                        <div class="h-64">
                            <BarChart
                                data={results.map((r) => r.forgetting)}
                                labels={results.map((r) =>
                                    r.method.toUpperCase(),
                                )}
                                label="Forgetting"
                                color="#EF4444"
                            />
                        </div>
                        <p class="text-xs text-gray-400 mt-2">
                            Lower is better
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Detailed Results Table -->
        <div class="card bg-base-300 shadow-xl">
            <div class="card-body">
                <h2 class="card-title text-2xl mb-4">Detailed Results</h2>

                <div class="overflow-x-auto">
                    <table class="table table-zebra">
                        <thead>
                            <tr>
                                <th>Method</th>
                                <th>Final Acc.</th>
                                <th>Avg Acc.</th>
                                <th>BWT</th>
                                <th>FWT</th>
                                <th>Forgetting</th>
                                <th>Intransigence</th>
                                <th>Time (s)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {#each results as result}
                                <tr>
                                    <td class="font-semibold"
                                        >{result.method.toUpperCase()}</td
                                    >
                                    <td>{result.finalAccuracy.toFixed(2)}%</td>
                                    <td>{result.averageAccuracy.toFixed(2)}%</td
                                    >
                                    <td
                                        class:text-success={result.bwt > 0}
                                        class:text-error={result.bwt < 0}
                                    >
                                        {result.bwt.toFixed(3)}
                                    </td>
                                    <td
                                        class:text-success={result.fwt > 0}
                                        class:text-error={result.fwt < 0}
                                    >
                                        {result.fwt.toFixed(3)}
                                    </td>
                                    <td
                                        class:text-success={result.forgetting <
                                            0.1}
                                        class:text-warning={result.forgetting >=
                                            0.1 && result.forgetting < 0.2}
                                        class:text-error={result.forgetting >=
                                            0.2}
                                    >
                                        {result.forgetting.toFixed(3)}
                                    </td>
                                    <td>{result.intransigence.toFixed(3)}</td>
                                    <td>{result.durationSeconds.toFixed(1)}</td>
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>

                <!-- Metric Explanations -->
                <div class="alert alert-info mt-6">
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
                    <div class="text-sm">
                        <h3 class="font-bold mb-1">Metric Definitions:</h3>
                        <ul class="list-disc list-inside space-y-1">
                            <li>
                                <strong>BWT (Backward Transfer)</strong>:
                                Measures forgetting. Positive = minimal
                                forgetting
                            </li>
                            <li>
                                <strong>FWT (Forward Transfer)</strong>:
                                Measures knowledge transfer to new tasks.
                                Positive = good
                            </li>
                            <li>
                                <strong>Forgetting</strong>: Average accuracy
                                drop on old tasks. Lower is better
                            </li>
                            <li>
                                <strong>Intransigence</strong>: Difficulty
                                learning new tasks. Lower is better
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    {/if}
</div>
