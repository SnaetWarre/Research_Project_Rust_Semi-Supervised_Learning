<script lang="ts">
  import { currentPage, modelInfo } from '$lib/stores/app';
  import { 
    LayoutDashboard, 
    GraduationCap, 
    ScanLine, 
    Tags, 
    PlayCircle, 
    Gauge,
    Leaf
  } from 'lucide-svelte';

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'training', label: 'Training', icon: GraduationCap },
    { id: 'inference', label: 'Inference', icon: ScanLine },
    { id: 'pseudo', label: 'Pseudo-Label', icon: Tags },
    { id: 'simulation', label: 'Simulation', icon: PlayCircle },
    { id: 'benchmark', label: 'Benchmark', icon: Gauge },
  ];

  function navigate(pageId: string) {
    currentPage.set(pageId);
  }
</script>

<aside class="w-64 h-screen bg-background-light border-r border-slate-700 flex flex-col">
  <!-- Logo/Header -->
  <div class="p-6 border-b border-slate-700">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
        <Leaf class="w-6 h-6 text-primary" />
      </div>
      <div>
        <h1 class="font-bold text-lg text-white">PlantVillage</h1>
        <p class="text-xs text-slate-400">SSL Dashboard</p>
      </div>
    </div>
  </div>

  <!-- Navigation -->
  <nav class="flex-1 p-4 overflow-y-auto">
    <ul class="space-y-2">
      {#each navItems as item}
        <li>
          <button
            class="w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors duration-200
              {$currentPage === item.id 
                ? 'bg-primary/20 text-primary' 
                : 'text-slate-400 hover:bg-slate-700/50 hover:text-white'}"
            onclick={() => navigate(item.id)}
          >
            <svelte:component this={item.icon} class="w-5 h-5" />
            <span class="font-medium">{item.label}</span>
          </button>
        </li>
      {/each}
    </ul>
  </nav>

  <!-- Status Footer -->
  <div class="p-4 border-t border-slate-700">
    <div class="bg-background rounded-lg p-3">
      <div class="flex items-center justify-between mb-2">
        <span class="text-xs text-slate-400">Model Status</span>
        <span class="w-2 h-2 rounded-full {$modelInfo.loaded ? 'bg-primary' : 'bg-slate-500'}"></span>
      </div>
      <p class="text-sm text-white truncate">
        {#if $modelInfo.loaded}
          {$modelInfo.path?.split('/').pop() || 'Loaded'}
        {:else}
          No model loaded
        {/if}
      </p>
    </div>
  </div>
</aside>
