<script lang="ts">
  import { open } from "@tauri-apps/plugin-dialog";
  import { className, listRuns, percent, type Run } from "./api";
  import LineChart from "./LineChart.svelte";

  let dir: string | null = $state(localStorage.getItem("runsDir"));
  let runs: Run[] = $state([]);
  let error: string | null = $state(null);

  async function refresh() {
    if (!dir) return;
    error = null;
    try {
      runs = await listRuns(dir);
    } catch (e) {
      error = String(e);
      runs = [];
    }
  }

  async function chooseDir() {
    const picked = await open({ directory: true, title: "Choose the runs folder" });
    if (typeof picked !== "string") return;
    dir = picked;
    localStorage.setItem("runsDir", picked);
    await refresh();
  }

  $effect(() => {
    refresh();
  });

  const weakest = (run: Run) => [...(run.test?.classes ?? [])].sort((a, b) => a.f1 - b.f1).slice(0, 3);
</script>

<section class="space-y-6">
  <div class="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-panel px-5 py-4">
    <div class="min-w-0">
      <p class="text-sm text-ink-3">Runs folder</p>
      <p class="truncate text-sm text-ink-2" title={dir ?? ""}>{dir ?? "None chosen. The CLI writes to runs/ by default."}</p>
    </div>
    <div class="flex gap-2">
      {#if dir}
        <button class="rounded-md px-3 py-1.5 text-sm text-ink-2 hover:bg-line" onclick={refresh}>Reload</button>
      {/if}
      <button class="rounded-md bg-leaf/15 px-3 py-1.5 text-sm text-leaf hover:bg-leaf/25" onclick={chooseDir}>Choose folder</button>
    </div>
  </div>

  {#if error}
    <p role="alert" class="rounded-md border border-red-400/30 bg-red-400/10 px-4 py-2 text-sm text-red-200">{error}</p>
  {:else if dir && runs.length === 0}
    <p class="text-sm text-ink-3">No model folders here yet.</p>
  {/if}

  {#each runs as run (run.name)}
    <article class="space-y-5 rounded-xl bg-panel p-5">
      <header class="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <h2 class="text-lg font-semibold">{run.name}</h2>
        {#if run.test}
          <dl class="flex gap-6 text-sm">
            <div><dt class="text-ink-3">Test accuracy</dt><dd class="text-xl font-semibold tabular-nums">{percent(run.test.accuracy, 2)}</dd></div>
            <div><dt class="text-ink-3">Macro F1</dt><dd class="text-xl font-semibold tabular-nums">{percent(run.test.macro_f1, 2)}</dd></div>
            <div><dt class="text-ink-3">Test images</dt><dd class="text-xl font-semibold tabular-nums">{run.test.images.toLocaleString()}</dd></div>
          </dl>
        {:else}
          <p class="text-sm text-ink-3">Not evaluated yet. Run <code>plantssl eval --model runs/{run.name}</code>.</p>
        {/if}
      </header>

      {#if run.train}
        <LineChart
          title="Validation accuracy per epoch"
          xLabel="epoch"
          points={run.train.epochs.map((e) => ({ x: e.epoch, y: e.val_accuracy }))}
          kept={run.train.best_epoch}
          format={(y) => percent(y)}
        />
      {/if}

      {#if run.ssl}
        <LineChart
          title="Validation accuracy per SSL round"
          xLabel="round"
          points={run.ssl.rounds.map((r) => ({ x: r.round, y: r.val_accuracy }))}
          kept={run.ssl.best_round}
          format={(y) => percent(y)}
        />
        <details class="text-sm">
          <summary class="cursor-pointer text-ink-2">Pseudo-labels per round</summary>
          <table class="mt-3 w-full text-left tabular-nums">
            <thead class="text-ink-3">
              <tr><th class="py-1 font-normal">Round</th><th class="font-normal">Revealed</th><th class="font-normal">Confident</th><th class="font-normal">Used</th><th class="font-normal">Correct</th><th class="font-normal">Validation</th></tr>
            </thead>
            <tbody class="text-ink-2">
              {#each run.ssl.rounds.slice(1) as r (r.round)}
                <tr class="border-t border-line">
                  <td class="py-1">{r.round}</td><td>{r.revealed.toLocaleString()}</td><td>{r.confident.toLocaleString()}</td>
                  <td>{r.accepted.toLocaleString()}</td><td>{percent(r.precision)}</td><td>{percent(r.val_accuracy, 2)}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </details>
      {/if}

      {#if run.test}
        <div class="text-sm">
          <p class="text-ink-3">Weakest classes on the test pool</p>
          <ul class="mt-1 space-y-0.5 text-ink-2">
            {#each weakest(run) as c (c.class)}
              <li class="flex justify-between gap-4"><span>{className(c.class)}</span><span class="tabular-nums">F1 {percent(c.f1)}</span></li>
            {/each}
          </ul>
        </div>
      {/if}
    </article>
  {/each}
</section>
