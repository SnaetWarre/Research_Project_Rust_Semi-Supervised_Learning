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
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #e5e7eb;
  }

  .card-title {
    font-size: 16px;
    font-weight: 600;
    color: #111827;
    margin: 0 0 16px 0;
  }
</style>
