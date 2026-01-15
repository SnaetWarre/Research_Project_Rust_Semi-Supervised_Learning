<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { OutputLine } from '$lib/stores/connection';
  
  export let lines: OutputLine[] = [];
  export let maxLines: number = 500;
  export let autoScroll: boolean = true;
  export let title: string = 'Terminal';
  
  let terminalEl: HTMLDivElement;
  let shouldAutoScroll = autoScroll;
  
  // Auto-scroll to bottom when new lines are added
  $: if (lines.length > 0 && shouldAutoScroll && terminalEl) {
    setTimeout(() => {
      terminalEl.scrollTop = terminalEl.scrollHeight;
    }, 10);
  }
  
  function handleScroll() {
    if (!terminalEl) return;
    // Disable auto-scroll if user scrolls up
    const isAtBottom = terminalEl.scrollHeight - terminalEl.scrollTop - terminalEl.clientHeight < 50;
    shouldAutoScroll = isAtBottom;
  }
  
  function scrollToBottom() {
    if (terminalEl) {
      terminalEl.scrollTop = terminalEl.scrollHeight;
      shouldAutoScroll = true;
    }
  }
  
  function formatTimestamp(timestamp: string): string {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('en-US', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return '';
    }
  }
</script>

<div class="terminal-container">
  <div class="terminal-header">
    <span class="terminal-title">{title}</span>
    <div class="terminal-controls">
      <span class="line-count">{lines.length} lines</span>
      {#if !shouldAutoScroll}
        <button class="scroll-btn" on:click={scrollToBottom} title="Scroll to bottom">
          ↓
        </button>
      {/if}
    </div>
  </div>
  
  <div 
    class="terminal" 
    bind:this={terminalEl}
    on:scroll={handleScroll}
  >
    {#if lines.length === 0}
      <div class="terminal-empty">No output yet...</div>
    {:else}
      {#each lines as line, i (i)}
        <div class="terminal-line {line.stream}">
          <span class="timestamp">{formatTimestamp(line.timestamp)}</span>
          <span class="content">{line.content}</span>
        </div>
      {/each}
    {/if}
  </div>
</div>

<style>
  .terminal-container {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #1a1a2e;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .terminal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: rgba(0, 0, 0, 0.3);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .terminal-title {
    font-size: 12px;
    font-weight: 600;
    color: rgba(255, 255, 255, 0.7);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .terminal-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .line-count {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.4);
  }
  
  .scroll-btn {
    background: rgba(255, 255, 255, 0.1);
    border: none;
    border-radius: 4px;
    color: white;
    padding: 2px 8px;
    font-size: 12px;
    cursor: pointer;
    transition: background 0.2s;
  }
  
  .scroll-btn:hover {
    background: rgba(255, 255, 255, 0.2);
  }
  
  .terminal {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
    line-height: 1.5;
  }
  
  .terminal-empty {
    color: rgba(255, 255, 255, 0.3);
    font-style: italic;
  }
  
  .terminal-line {
    display: flex;
    gap: 8px;
    margin-bottom: 2px;
  }
  
  .terminal-line.stdout .content {
    color: #e0e0e0;
  }
  
  .terminal-line.stderr .content {
    color: #ff6b6b;
  }
  
  .timestamp {
    color: rgba(255, 255, 255, 0.3);
    flex-shrink: 0;
  }
  
  .content {
    white-space: pre-wrap;
    word-break: break-word;
  }
  
  /* Scrollbar styling */
  .terminal::-webkit-scrollbar {
    width: 8px;
  }
  
  .terminal::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
  }
  
  .terminal::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
  }
  
  .terminal::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
  }
</style>
