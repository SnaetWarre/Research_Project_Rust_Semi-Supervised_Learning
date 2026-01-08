<script lang="ts">
    import "../app.css";
    import Sidebar from "$lib/components/Sidebar.svelte";
    import { currentPage, datasetInfo, addActivity } from "$lib/stores/app";
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
                        `Dataset auto-loaded: ${result.total_samples} samples, ${result.num_classes} classes`,
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
                            `Dataset loaded from fallback location: ${result.total_samples} samples`,
                        );
                    } catch (e2) {
                        addActivity(
                            "info",
                            `No dataset auto-loaded. You can load one from the Dashboard.`,
                        );
                    }
                }
            }
        }, 100);
    });
</script>

{#if appReady}
    <div
        class="flex h-screen overflow-hidden bg-background"
        style="background-color: #0F172A;"
    >
        <Sidebar />

        <main class="flex-1 overflow-y-auto">
            <CurrentPage />
        </main>
    </div>
{:else}
    <div
        class="flex h-screen items-center justify-center"
        style="background-color: #0F172A; display: flex; height: 100vh; align-items: center; justify-content: center;"
    >
        <div style="text-align: center;">
            <div
                style="width: 64px; height: 64px; border: 4px solid #10B981; border-top-color: transparent; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 16px;"
            ></div>
            <p style="color: #94A3B8; font-family: sans-serif;">
                Loading PlantVillage SSL Dashboard...
            </p>
        </div>
    </div>
{/if}

<!-- SvelteKit requires this but we don't use it -->
{@render children?.()}

<style>
    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
</style>
