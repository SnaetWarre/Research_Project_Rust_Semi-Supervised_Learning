<script lang="ts">
  interface Props {
    value: number;
    label: string;
    showPercentage?: boolean;
  }

  let { value, label, showPercentage = true }: Props = $props();

  // Color based on confidence
  const getColor = (v: number) => {
    if (v >= 0.9) return 'bg-emerald-500';
    if (v >= 0.7) return 'bg-yellow-500';
    return 'bg-red-500';
  };
</script>

<div class="space-y-1">
  <div class="flex justify-between text-sm">
    <span class="text-slate-300 truncate max-w-[200px]" title={label}>{label}</span>
    {#if showPercentage}
      <span class="text-white font-medium">{(value * 100).toFixed(1)}%</span>
    {/if}
  </div>
  <div class="h-2 bg-background-lighter rounded-full overflow-hidden">
    <div
      class="{getColor(value)} h-full rounded-full transition-all duration-300"
      style="width: {value * 100}%"
    ></div>
  </div>
</div>
