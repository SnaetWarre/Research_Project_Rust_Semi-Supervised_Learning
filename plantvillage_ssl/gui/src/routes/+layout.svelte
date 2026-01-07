<script lang="ts">
  import '../app.css';
  import Sidebar from '$lib/components/Sidebar.svelte';
  import { currentPage } from '$lib/stores/app';
  
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
</script>

<div class="flex h-screen overflow-hidden bg-background">
  <Sidebar />
  
  <main class="flex-1 overflow-y-auto">
    <CurrentPage />
  </main>
</div>

<!-- SvelteKit requires this but we don't use it -->
{@render children?.()}
