<script lang="ts">
  import Classify from "./lib/Classify.svelte";
  import Runs from "./lib/Runs.svelte";

  const tabs = [
    { id: "classify", label: "Classify" },
    { id: "runs", label: "Runs" },
  ] as const;

  let tab: (typeof tabs)[number]["id"] = $state("classify");
</script>

<div class="min-h-dvh">
  <header class="flex items-center gap-6 border-b border-line px-6 py-3">
    <h1 class="text-base font-semibold tracking-tight">
      <span class="text-leaf">Plant</span>SSL
    </h1>
    <nav class="flex gap-1" aria-label="Sections">
      {#each tabs as t (t.id)}
        <button
          class="rounded-md px-3 py-1.5 text-sm transition-colors {tab === t.id
            ? 'bg-panel text-ink'
            : 'text-ink-3 hover:text-ink-2'}"
          aria-current={tab === t.id ? "page" : undefined}
          onclick={() => (tab = t.id)}>{t.label}</button
        >
      {/each}
    </nav>
  </header>

  <main class="mx-auto max-w-5xl p-6">
    <!-- Both stay mounted so a loaded model or run list survives switching tabs -->
    <div hidden={tab !== "classify"}><Classify /></div>
    <div hidden={tab !== "runs"}><Runs /></div>
  </main>
</div>
