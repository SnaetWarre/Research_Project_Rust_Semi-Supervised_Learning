<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Chart from 'chart.js/auto';

  interface Props {
    data: number[];
    labels?: string[];
    label?: string;
    color?: string;
    yAxisLabel?: string;
    showArea?: boolean;
  }

  let { data, labels, label = 'Value', color = '#10B981', yAxisLabel, showArea = true }: Props = $props();

  let canvas: HTMLCanvasElement;
  let chart: Chart | null = null;

  function createChart() {
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const chartLabels = labels || data.map((_, i) => String(i + 1));

    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: chartLabels,
        datasets: [{
          label,
          data,
          borderColor: color,
          backgroundColor: showArea ? `${color}20` : 'transparent',
          fill: showArea,
          tension: 0.4,
          pointRadius: data.length > 50 ? 0 : 3,
          pointHoverRadius: 5,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: false,
          },
        },
        scales: {
          x: {
            grid: {
              color: '#334155',
            },
            ticks: {
              color: '#94A3B8',
            },
          },
          y: {
            grid: {
              color: '#334155',
            },
            ticks: {
              color: '#94A3B8',
            },
            title: yAxisLabel ? {
              display: true,
              text: yAxisLabel,
              color: '#94A3B8',
            } : undefined,
          },
        },
        interaction: {
          intersect: false,
          mode: 'index',
        },
      },
    });
  }

  function updateChart() {
    if (!chart) return;

    const chartLabels = labels || data.map((_, i) => String(i + 1));
    chart.data.labels = chartLabels;
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
