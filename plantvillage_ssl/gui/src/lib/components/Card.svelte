<script lang="ts">
  interface Props {
    title?: string;
    class?: string;
    children?: import('svelte').Snippet;
  }

  let { title, class: className = '', children }: Props = $props();
</script>

<div class="bg-background-light rounded-xl p-6 shadow-lg {className}">
  {#if title}
    <h3 class="text-lg font-semibold text-white mb-4">{title}</h3>
  {/if}
  {@render children?.()}
</div>
