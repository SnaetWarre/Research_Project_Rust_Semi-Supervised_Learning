<script lang="ts">
  interface Props {
    title?: string;
    class?: string;
    children?: import('svelte').Snippet;
  }

  let { title, class: className = '', children }: Props = $props();
</script>

<div class="card-component {className}">
  {#if title}
    <h3 class="card-title">{title}</h3>
  {/if}
  {@render children?.()}
</div>

<style>
  .card-component {
    background-color: var(--bg-panel);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid var(--border-base);
  }

  .card-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-main);
    margin: 0 0 16px 0;
  }
</style>
