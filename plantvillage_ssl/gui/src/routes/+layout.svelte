<script lang="ts">
    import "../app.css";
    import Sidebar from "$lib/components/Sidebar.svelte";
    import TitleBar from "$lib/components/TitleBar.svelte";
    import { currentPage, datasetInfo, modelInfo, addActivity } from "$lib/stores/app";
    import { invoke } from "@tauri-apps/api/core";
    import { onMount } from "svelte";

    import Dashboard from "$lib/pages/Dashboard.svelte";
    import Training from "$lib/pages/Training.svelte";
    import Inference from "$lib/pages/Inference.svelte";
    import PseudoLabel from "$lib/pages/PseudoLabel.svelte";
    import Simulation from "$lib/pages/Simulation.svelte";
    import Benchmark from "$lib/pages/Benchmark.svelte";
    import IncrementalLearning from "$lib/pages/IncrementalLearning.svelte";
    import Experiment from "$lib/pages/Experiment.svelte";
    import Diagnostics from "$lib/pages/Diagnostics.svelte";

    import type { Snippet } from "svelte";

    interface Props {
        children: Snippet;
    }

    let { children }: Props = $props();
    let appReady = $state(false);

    const pages: Record<string, typeof Dashboard> = {
        dashboard: Dashboard,
        training: Training,
        inference: Inference,
        diagnostics: Diagnostics,
        pseudo: PseudoLabel,
        simulation: Simulation,
        benchmark: Benchmark,
        incremental: IncrementalLearning,
        experiment: Experiment,
    };

    // Get current page component
    const CurrentPage = $derived(pages[$currentPage] || Dashboard);

    // Auto-load balanced dataset on startup (non-blocking)
    onMount(() => {
        // Mark app as ready immediately
        appReady = true;

        // Defer dataset loading to not block UI rendering
        setTimeout(async () => {
            // Only load if not already loaded
            if (!$datasetInfo) {
                try {
                    const defaultDataDir = "data/plantvillage/balanced";
                    const result = await invoke<{
                        path: string;
                        total_samples: number;
                        num_classes: number;
                        class_names: string[];
                        class_counts: number[];
                    }>("get_dataset_stats", { dataDir: defaultDataDir });

                    datasetInfo.set({
                        path: result.path,
                        totalSamples: result.total_samples,
                        numClasses: result.num_classes,
                        classNames: result.class_names,
                        classCounts: result.class_counts,
                    });
                    addActivity(
                        "success",
                        `Dataset loaded: ${result.total_samples} samples, ${result.num_classes} classes`,
                    );
                } catch (e) {
                    // Try fallback to regular data directory
                    try {
                        const fallbackDataDir = "data/plantvillage";
                        const result = await invoke<{
                            path: string;
                            total_samples: number;
                            num_classes: number;
                            class_names: string[];
                            class_counts: number[];
                        }>("get_dataset_stats", { dataDir: fallbackDataDir });

                        datasetInfo.set({
                            path: result.path,
                            totalSamples: result.total_samples,
                            numClasses: result.num_classes,
                            classNames: result.class_names,
                            classCounts: result.class_counts,
                        });
                        addActivity(
                            "warning",
                            `Dataset loaded from fallback: ${result.total_samples} samples`,
                        );
                    } catch (e2) {
                        addActivity(
                            "info",
                            `No dataset auto-loaded. Load one from Dashboard.`,
                        );
                    }
                }
            }

            // Auto-load model if not loaded
            if (!$modelInfo.loaded) {
                try {
                    const result = await invoke<any>("load_model", { modelPath: "best_model.mpk" });
                    modelInfo.set({
                        loaded: result.loaded,
                        path: result.path,
                        numClasses: result.num_classes,
                        inputSize: result.input_size,
                    });
                    addActivity("success", `Auto-loaded model: ${result.path}`);
                } catch (e) {
                    // Silent fail for model
                }
            }
        }, 100);
    });
</script>

{#if appReady}
    <TitleBar />
    <div class="app-container">
        <Sidebar />
        <main class="main-content">
            <CurrentPage />
        </main>
    </div>
{:else}
    <TitleBar />
    <div class="loading-screen">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <p class="loading-text">Loading PlantVillage SSL...</p>
        </div>
    </div>
{/if}

<!-- SvelteKit requires this but we don't use it -->
{@render children?.()}

<style>
    .app-container {
        display: flex;
        height: calc(100vh - 32px);
        margin-top: 32px;
        overflow: hidden;
        background-color: var(--bg-app);
    }

    .main-content {
        flex: 1;
        overflow-y: auto;
        padding: 24px;
    }

    .loading-screen {
        display: flex;
        height: calc(100vh - 32px);
        margin-top: 32px;
        align-items: center;
        justify-content: center;
        background-color: var(--bg-app);
    }

    .loading-content {
        text-align: center;
    }

    .loading-spinner {
        width: 48px;
        height: 48px;
        border: 3px solid var(--c-zinc-800);
        border-top-color: var(--c-accent);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin: 0 auto 16px;
    }

    .loading-text {
        color: var(--text-secondary);
        font-family: "Inter", sans-serif;
        font-size: 14px;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
</style>
