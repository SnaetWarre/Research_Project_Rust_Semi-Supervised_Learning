<script lang="ts">
  import { listen } from "@tauri-apps/api/event";
  import { open } from "@tauri-apps/plugin-dialog";
  import { onDestroy } from "svelte";
  import { percent, startSsl, stopSsl, type RoundLog, type SslEvent } from "./api";
  import LineChart from "./LineChart.svelte";

  let split = $state(localStorage.getItem("liveSplit") ?? "");
  let init = $state(localStorage.getItem("liveInit") ?? "");
  let out = $state(localStorage.getItem("liveOut") ?? "");
  let rounds = $state(3);
  let keep = $state(0.5);
  let mode: "student" | "fine-tune" = $state("student");
  let epochsPerRound = $state(30);

  let running = $state(false);
  let stopping = $state(false);
  let history: RoundLog[] = $state([]);
  let pass = $state({ label: "", images: 0, done: 0 });
  let lastEpoch: string | null = $state(null);
  let outcome: { tone: "ok" | "error"; text: string } | null = $state(null);

  const currentRound = $derived(history.length === 0 ? 0 : history[history.length - 1].round + 1);
  const best = $derived(history.reduce<RoundLog | null>((a, b) => (!a || b.val_accuracy > a.val_accuracy ? b : a), null));

  const unlisten = listen<SslEvent>("ssl", ({ payload: e }) => {
    switch (e.kind) {
      case "pass":
        pass = { label: e.label, images: e.images, done: 0 };
        break;
      case "progress":
        pass.done = e.done;
        break;
      case "epoch":
        lastEpoch = `epoch ${e.epoch}: validation ${percent(e.val_accuracy, 2)}`;
        break;
      case "round":
        history.push(e);
        lastEpoch = null;
        break;
      case "finished":
        finish({ tone: "ok", text: `Done. Kept round ${e.best_round} and saved it to ${e.out}.` });
        break;
      case "stopped":
        finish({ tone: "ok", text: "Stopped. Nothing was saved." });
        break;
      case "failed":
        finish({ tone: "error", text: e.message });
        break;
    }
  });
  onDestroy(() => unlisten.then((stop) => stop()));

  function finish(result: typeof outcome) {
    running = false;
    stopping = false;
    outcome = result;
  }

  async function pick(kind: "split" | "init" | "out") {
    const picked =
      kind === "split"
        ? await open({ title: "Choose split.json", filters: [{ name: "Split", extensions: ["json"] }] })
        : await open({ directory: true, title: kind === "init" ? "Choose the starting model" : "Choose where to save" });
    if (typeof picked !== "string") return;
    if (kind === "split") split = picked;
    if (kind === "init") init = picked;
    if (kind === "out") out = picked;
  }

  async function start() {
    localStorage.setItem("liveSplit", split);
    localStorage.setItem("liveInit", init);
    localStorage.setItem("liveOut", out);
    history = [];
    outcome = null;
    lastEpoch = null;
    pass = { label: "loading", images: 0, done: 0 };
    running = true;
    try {
      await startSsl({ split, init, out, rounds, keep, mode, epochs_per_round: epochsPerRound });
    } catch (e) {
      finish({ tone: "error", text: String(e) });
    }
  }

  async function stop() {
    stopping = true;
    await stopSsl();
  }

  const paths = [
    { key: "split", label: "Split file", hint: "runs/split.json from plantssl split" },
    { key: "init", label: "Starting model", hint: "a folder from plantssl train" },
    { key: "out", label: "Save to", hint: "an empty folder, for example runs/ssl-live" },
  ] as const;
  const values = $derived({ split, init, out });
</script>

<section class="space-y-6">
  <div class="space-y-4 rounded-xl bg-panel p-5">
    <p class="text-sm text-ink-2">
      Unlabeled photos arrive in rounds. Each round the current model labels them, keeps its most confident labels for
      every disease, and a model trains on them. Watch validation accuracy change as the rounds go by.
    </p>
    <div class="grid gap-3">
      {#each paths as p (p.key)}
        <div class="flex items-center gap-3">
          <span class="w-32 shrink-0 text-sm text-ink-3">{p.label}</span>
          <span class="min-w-0 flex-1 truncate text-sm {values[p.key] ? 'text-ink-2' : 'text-ink-3'}" title={values[p.key]}>
            {values[p.key] || p.hint}
          </span>
          <button class="rounded-md px-3 py-1 text-sm text-ink-2 hover:bg-line disabled:opacity-40" disabled={running} onclick={() => pick(p.key)}>
            Choose
          </button>
        </div>
      {/each}
    </div>
    <div class="flex flex-wrap items-end gap-4 text-sm">
      <label class="grid gap-1"><span class="text-ink-3">Rounds</span>
        <input type="number" min="1" max="50" bind:value={rounds} disabled={running} class="w-20 rounded-md bg-surface px-2 py-1 ring-1 ring-line" /></label>
      <label class="grid gap-1"><span class="text-ink-3">Keep per class</span>
        <input type="number" min="0.05" max="1" step="0.05" bind:value={keep} disabled={running} class="w-24 rounded-md bg-surface px-2 py-1 ring-1 ring-line" /></label>
      <label class="grid gap-1"><span class="text-ink-3">Epochs per round</span>
        <input type="number" min="1" max="50" bind:value={epochsPerRound} disabled={running} class="w-24 rounded-md bg-surface px-2 py-1 ring-1 ring-line" /></label>
      <label class="grid gap-1"><span class="text-ink-3">Each round</span>
        <select bind:value={mode} disabled={running} onchange={() => (epochsPerRound = mode === "student" ? 30 : 3)}
          class="rounded-md bg-surface px-2 py-1 ring-1 ring-line">
          <option value="student">Train a new model (slow, best)</option>
          <option value="fine-tune">Fine-tune the current model (fast)</option>
        </select></label>
      <div class="ml-auto">
        {#if running}
          <button class="rounded-md bg-red-400/15 px-4 py-1.5 text-red-200 hover:bg-red-400/25 disabled:opacity-50" disabled={stopping} onclick={stop}>
            {stopping ? "Stopping..." : "Stop"}
          </button>
        {:else}
          <button class="rounded-md bg-leaf/15 px-4 py-1.5 text-leaf hover:bg-leaf/25 disabled:opacity-40" disabled={!split || !init || !out} onclick={start}>
            Start
          </button>
        {/if}
      </div>
    </div>
  </div>

  {#if outcome}
    <p role="status" class="rounded-md px-4 py-2 text-sm {outcome.tone === 'ok' ? 'bg-leaf/10 text-leaf' : 'border border-red-400/30 bg-red-400/10 text-red-200'}">
      {outcome.text}
    </p>
  {/if}

  {#if running || history.length > 0}
    <div class="space-y-5 rounded-xl bg-panel p-5">
      {#if running}
        <div aria-live="polite">
          <div class="flex justify-between gap-3 text-sm">
            <span class="text-ink-2">
              {currentRound === 0 ? "Measuring the starting model" : `Round ${currentRound} of ${rounds}`}: {pass.label}
            </span>
            <span class="tabular-nums text-ink-3">{pass.done.toLocaleString()} / {pass.images.toLocaleString()} images</span>
          </div>
          <div class="mt-2 h-1.5 rounded-full bg-line">
            <div class="h-full rounded-full bg-series transition-[width]" style:width={pass.images ? `${(100 * pass.done) / pass.images}%` : "0%"}></div>
          </div>
          {#if lastEpoch}<p class="mt-2 text-xs text-ink-3">Last {lastEpoch}</p>{/if}
        </div>
      {/if}

      {#if best}
        <dl class="flex flex-wrap gap-8 text-sm">
          <div><dt class="text-ink-3">Start</dt><dd class="text-xl font-semibold tabular-nums">{percent(history[0].val_accuracy, 2)}</dd></div>
          <div><dt class="text-ink-3">Best so far</dt><dd class="text-xl font-semibold tabular-nums">{percent(best.val_accuracy, 2)}</dd></div>
          <div><dt class="text-ink-3">Pseudo-labels in use</dt><dd class="text-xl font-semibold tabular-nums">{history[history.length - 1].accepted.toLocaleString()}</dd></div>
        </dl>
      {/if}

      {#if history.length > 1}
        <LineChart
          title="Validation accuracy per round"
          xLabel="round"
          points={history.map((r) => ({ x: r.round, y: r.val_accuracy }))}
          kept={best?.round}
          format={(y) => percent(y)}
        />
        <table class="w-full text-left text-sm tabular-nums">
          <thead class="text-ink-3">
            <tr><th class="py-1 font-normal">Round</th><th class="font-normal">Seen</th><th class="font-normal">Used</th><th class="font-normal">Correct</th><th class="font-normal">Validation</th></tr>
          </thead>
          <tbody class="text-ink-2">
            {#each history.slice(1) as r (r.round)}
              <tr class="border-t border-line">
                <td class="py-1">{r.round}</td><td>{r.revealed.toLocaleString()}</td><td>{r.accepted.toLocaleString()}</td>
                <td>{percent(r.precision)}</td><td>{percent(r.val_accuracy, 2)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
  {/if}
</section>
