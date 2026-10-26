<script lang="ts">
  interface Point {
    x: number;
    y: number;
  }

  interface Props {
    title: string;
    points: Point[];
    xLabel: string;
    format: (y: number) => string;
    /** x of the point the run kept, marked on the line. */
    kept?: number;
  }

  let { title, points, xLabel, format, kept }: Props = $props();

  const height = 180;
  const pad = { top: 12, right: 16, bottom: 28, left: 52 };
  let width = $state(480);
  let hover: Point | null = $state(null);

  const xs = $derived(points.map((p) => p.x));
  const ys = $derived(points.map((p) => p.y));
  const xMin = $derived(Math.min(...xs));
  const xMax = $derived(Math.max(...xs));
  const yMin = $derived(Math.max(0, Math.min(...ys) - 0.02));
  const yMax = $derived(Math.min(1, Math.max(...ys) + 0.02));

  const sx = (x: number) => pad.left + ((x - xMin) / Math.max(xMax - xMin, 1)) * (width - pad.left - pad.right);
  const sy = (y: number) => pad.top + (1 - (y - yMin) / Math.max(yMax - yMin, 1e-9)) * (height - pad.top - pad.bottom);

  const path = $derived(points.map((p, i) => `${i ? "L" : "M"}${sx(p.x)},${sy(p.y)}`).join(""));
  const ticks = $derived([0, 1, 2, 3].map((i) => yMin + ((yMax - yMin) * i) / 3));
  const keptPoint = $derived(points.find((p) => p.x === kept));

  function onMove(event: PointerEvent) {
    const box = (event.currentTarget as SVGElement).getBoundingClientRect();
    const x = event.clientX - box.left;
    hover = points.reduce((a, b) => (Math.abs(sx(b.x) - x) < Math.abs(sx(a.x) - x) ? b : a));
  }
</script>

<figure class="space-y-2">
  <figcaption class="text-sm text-ink-2">{title}</figcaption>
  <div class="relative" bind:clientWidth={width}>
    <svg
      {width}
      {height}
      role="img"
      aria-label="{title}, {points.length} points"
      onpointermove={onMove}
      onpointerleave={() => (hover = null)}
    >
      {#each ticks as t (t)}
        <line x1={pad.left} x2={width - pad.right} y1={sy(t)} y2={sy(t)} class="stroke-line" />
        <text x={pad.left - 8} y={sy(t)} dy="0.32em" text-anchor="end" class="fill-ink-3 text-[11px] tabular-nums"
          >{format(t)}</text
        >
      {/each}
      <text x={width - pad.right} y={height - 6} text-anchor="end" class="fill-ink-3 text-[11px]">{xLabel}</text>
      <text x={pad.left} y={height - 6} class="fill-ink-3 text-[11px] tabular-nums">{xMin}</text>

      <path d={path} fill="none" class="stroke-series" stroke-width="2" stroke-linejoin="round" />
      {#if keptPoint}
        <circle cx={sx(keptPoint.x)} cy={sy(keptPoint.y)} r="4.5" class="fill-series stroke-panel" stroke-width="2" />
      {/if}
      {#if hover}
        <line x1={sx(hover.x)} x2={sx(hover.x)} y1={pad.top} y2={height - pad.bottom} class="stroke-ink-3" />
        <circle cx={sx(hover.x)} cy={sy(hover.y)} r="4" class="fill-series stroke-panel" stroke-width="2" />
      {/if}
    </svg>
    {#if hover}
      <div
        class="pointer-events-none absolute top-0 rounded-md bg-surface px-2.5 py-1.5 text-xs shadow-lg ring-1 ring-line"
        style:left="{Math.min(sx(hover.x) + 10, width - 120)}px"
      >
        <strong class="block text-ink tabular-nums">{format(hover.y)}</strong>
        <span class="text-ink-3">{xLabel} {hover.x}{hover.x === kept ? " (kept)" : ""}</span>
      </div>
    {/if}
  </div>
</figure>
