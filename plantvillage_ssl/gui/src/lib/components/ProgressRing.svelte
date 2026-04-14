<script lang="ts">
  interface Props {
    value: number;
    max?: number;
    size?: number;
    strokeWidth?: number;
    label?: string;
  }

  let { value, max = 100, size = 120, strokeWidth = 8, label }: Props = $props();

  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = $derived(circumference - (value / max) * circumference);
</script>

<div class="relative inline-flex items-center justify-center" style="width: {size}px; height: {size}px">
  <svg class="transform -rotate-90" width={size} height={size}>
    <!-- Background circle -->
    <circle
      stroke="var(--c-zinc-800)"
      stroke-width={strokeWidth}
      fill="transparent"
      r={radius}
      cx={size / 2}
      cy={size / 2}
    />
    <!-- Progress circle -->
    <circle
      stroke="var(--c-accent)"
      stroke-width={strokeWidth}
      stroke-linecap="round"
      fill="transparent"
      r={radius}
      cx={size / 2}
      cy={size / 2}
      style="stroke-dasharray: {circumference}; stroke-dashoffset: {offset}; transition: stroke-dashoffset 0.5s ease-out;"
    />
  </svg>
  <div class="absolute flex flex-col items-center justify-center">
    <span class="text-2xl font-bold" style="color: var(--text-main);">{value.toFixed(1)}%</span>
    {#if label}
      <span class="text-xs" style="color: var(--text-secondary);">{label}</span>
    {/if}
  </div>
</div>
