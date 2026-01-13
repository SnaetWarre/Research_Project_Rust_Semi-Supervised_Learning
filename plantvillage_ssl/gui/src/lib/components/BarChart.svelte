<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Chart from 'chart.js/auto';

  interface Props {
    data: number[];
    labels: string[];
    label?: string;
    color?: string;
    horizontal?: boolean;
  }

  let { data, labels, label = 'Value', color = '#10B981', horizontal = false }: Props = $props();

  let canvas: HTMLCanvasElement;
  let chart: Chart | null = null;

  function createChart() {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label,
          data,
          backgroundColor: `${color}80`,
          borderColor: color,
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: horizontal ? 'y' : 'x',
        plugins: {
          legend: {
            display: false,
          },
        },
          scales: {
          x: {
            grid: {
              color: '#27272a',
            },
            ticks: {
              color: '#a1a1aa',
            },
          },
          y: {
            grid: {
              color: '#27272a',
            },
            ticks: {
              color: '#a1a1aa',
            },
          },
        },
      },
    });
  }

  function updateChart() {
    if (!chart) return;

    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update('none');
  }

  onMount(() => {
    createChart();
  });

  onDestroy(() => {
    if (chart) {
      chart.destroy();
    }
  });

  $effect(() => {
    if (data && chart) {
      updateChart();
    }
  });
</script>

<div class="h-full w-full">
  <canvas bind:this={canvas}></canvas>
</div>
