<script lang="ts">
  import '../app.css';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import { currentPage, datasetInfo, addActivity } from '$lib/stores/app';
  import { invoke } from '@tauri-apps/api/core';
  import { onMount } from 'svelte';
  
  import Dashboard from '$lib/pages/Dashboard.svelte';
  import Training from '$lib/pages/Training.svelte';
  import Inference from '$lib/pages/Inference.svelte';
  import PseudoLabel from '$lib/pages/PseudoLabel.svelte';
  import Simulation from '$lib/pages/Simulation.svelte';
  import Benchmark from '$lib/pages/Benchmark.svelte';

  import type { Snippet } from 'svelte';
  
  interface Props {
    children: Snippet;
  }
  
  let { children }: Props = $props();

  const pages: Record<string, typeof Dashboard> = {
    dashboard: Dashboard,
    training: Training,
    inference: Inference,
    pseudo: PseudoLabel,
    simulation: Simulation,
    benchmark: Benchmark,
  };
  
  // Get current page component
  const CurrentPage = $derived(pages[$currentPage] || Dashboard);

  // Auto-load balanced dataset on startup
  onMount(async () => {
    // Only load if not already loaded
    if (!$datasetInfo) {
      try {
        const defaultDataDir = 'data/plantvillage/balanced';
        const result = await invoke<{
          path: string;
          total_samples: number;
          num_classes: number;
          class_names: string[];
          class_counts: number[];
        }>('get_dataset_stats', { dataDir: defaultDataDir });

        datasetInfo.set({
          path: result.path,
          totalSamples: result.total_samples,
          numClasses: result.num_classes,
          classNames: result.class_names,
          classCounts: result.class_counts,
        });
        addActivity('success', `Dataset auto-loaded: ${result.total_samples} samples, ${result.num_classes} classes`);
      } catch (e) {
        // Try fallback to regular data directory
        try {
          const fallbackDataDir = 'data/plantvillage';
          const result = await invoke<{
            path: string;
            total_samples: number;
            num_classes: number;
            class_names: string[];
            class_counts: number[];
          }>('get_dataset_stats', { dataDir: fallbackDataDir });

          datasetInfo.set({
            path: result.path,
            totalSamples: result.total_samples,
            numClasses: result.num_classes,
            classNames: result.class_names,
            classCounts: result.class_counts,
          });
          addActivity('warning', `Dataset loaded from fallback location: ${result.total_samples} samples`);
        } catch (e2) {
          addActivity('error', `Failed to auto-load dataset: ${e2}. Please ensure dataset exists at data/plantvillage/balanced`);
        }
      }
    }
  });
</script>

<div class="flex h-screen overflow-hidden bg-background">
  <Sidebar />
  
  <main class="flex-1 overflow-y-auto">
    <CurrentPage />
  </main>
</div>

<!-- SvelteKit requires this but we don't use it -->
{@render children?.()}
