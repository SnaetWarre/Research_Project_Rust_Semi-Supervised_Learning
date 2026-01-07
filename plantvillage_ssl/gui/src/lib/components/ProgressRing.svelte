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
      class="text-background-lighter"
      stroke="currentColor"
      stroke-width={strokeWidth}
      fill="transparent"
      r={radius}
      cx={size / 2}
      cy={size / 2}
    />
    <!-- Progress circle -->
    <circle
      class="text-primary transition-all duration-500 ease-out"
      stroke="currentColor"
      stroke-width={strokeWidth}
      stroke-linecap="round"
      fill="transparent"
      r={radius}
      cx={size / 2}
      cy={size / 2}
      style="stroke-dasharray: {circumference}; stroke-dashoffset: {offset}"
    />
  </svg>
  <div class="absolute flex flex-col items-center justify-center">
    <span class="text-2xl font-bold text-white">{value.toFixed(1)}%</span>
    {#if label}
      <span class="text-xs text-slate-400">{label}</span>
    {/if}
  </div>
</div>
