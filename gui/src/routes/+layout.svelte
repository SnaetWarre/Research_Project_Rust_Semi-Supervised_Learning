<script lang="ts">
    import "../app.css";
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
    import InteractiveSSL from "$lib/pages/InteractiveSSL.svelte";

    import type { Snippet } from "svelte";

    interface Props {
        children: Snippet;
    }

    let { children }: Props = $props();
    let appReady = $state(false);

    const pages: Record<string, typeof Dashboard> = {
        dashboard: Dashboard,
        demo: InteractiveSSL,
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
        }, 100);
    });

    function setPage(page: string) {
        currentPage.set(page);
    }
</script>

{#if appReady}
    <div class="app-container">
        <main class="main-content">
            <CurrentPage />
        </main>
        
        <!-- Mobile Bottom Navigation -->
        <nav class="mobile-nav">
            <button class:active={$currentPage === 'dashboard'} onclick={() => setPage('dashboard')}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>
                    <polyline points="9 22 9 12 15 12 15 22"></polyline>
                </svg>
                <span>Home</span>
            </button>
            <button class:active={$currentPage === 'demo'} onclick={() => setPage('demo')}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v20M2 12h20"></path>
                    <path d="M12 2l-5 5M12 2l5 5M12 22l-5-5M12 22l5-5"></path>
                </svg>
                <span>Demo</span>
            </button>
            <button class:active={$currentPage === 'diagnostics'} onclick={() => setPage('diagnostics')}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="8" x2="12" y2="12"></line>
                    <line x1="12" y1="16" x2="12.01" y2="16"></line>
                </svg>
                <span>Diag</span>
            </button>
            <button class:active={$currentPage === 'simulation'} onclick={() => setPage('simulation')}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                    <line x1="8" y1="21" x2="16" y2="21"></line>
                    <line x1="12" y1="17" x2="12" y2="21"></line>
                </svg>
                <span>Sim</span>
            </button>
        </nav>
    </div>
{:else}
    <div class="loading-screen">
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <p class="loading-text">Loading...</p>
        </div>
    </div>
{/if}

<!-- SvelteKit requires this but we don't use it -->
{@render children?.()}

<style lang="css">
    :global(body) {
        margin: 0;
        padding: 0;
        background-color: #000; /* iOS dark mode standard */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    .app-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        width: 100vw;
        overflow: hidden;
        background-color: var(--bg-app);
    }

    .main-content {
        flex: 1;
        overflow-y: auto;
        padding: env(safe-area-inset-top) 16px 80px 16px; /* Bottom padding for nav bar */
        -webkit-overflow-scrolling: touch;
    }

    /* Mobile Bottom Navigation */
    .mobile-nav {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 60px;
        background: rgba(20, 20, 20, 0.95); /* Glassmorphism */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding-bottom: env(safe-area-inset-bottom);
        z-index: 1000;
    }

    .mobile-nav button {
        background: none;
        border: none;
        color: #888;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4px;
        flex: 1;
        height: 100%;
        font-size: 10px;
        font-weight: 500;
    }

    .mobile-nav button.active {
        color: var(--c-accent, #10b981);
    }

    .mobile-nav button svg {
        width: 24px;
        height: 24px;
        margin-bottom: 4px;
    }

    .loading-screen {
        display: flex;
        height: 100vh;
        align-items: center;
        justify-content: center;
        background-color: var(--bg-app);
    }

    .loading-content {
        text-align: center;
    }

    .loading-spinner {
        width: 40px;
        height: 40px;
        border: 3px solid rgba(255, 255, 255, 0.1);
        border-top-color: var(--c-accent, #10b981);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin: 0 auto 16px;
    }

    .loading-text {
        color: #888;
        font-size: 14px;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }
</style>
