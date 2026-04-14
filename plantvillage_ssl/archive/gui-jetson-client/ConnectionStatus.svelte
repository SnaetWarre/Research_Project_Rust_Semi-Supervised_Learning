<script lang="ts">
  import { connectionStore, isConnected, isRemote } from '$lib/stores/connection';
  import type { ConnectionState } from '$lib/stores/connection';
  
  export let compact: boolean = false;
  
  let state: ConnectionState;
  connectionStore.subscribe(s => state = s);
  
  let showDropdown = false;
  let urlInput = '';
  
  $: urlInput = state.jetsonUrl;
  
  async function handleConnect() {
    try {
      await connectionStore.connect(urlInput);
      showDropdown = false;
    } catch (e) {
      console.error('Connection failed:', e);
    }
  }
  
  async function handleDisconnect() {
    await connectionStore.disconnect();
    showDropdown = false;
  }
  
  async function handleTest() {
    await connectionStore.testConnection();
  }
  
  function getStatusColor(): string {
    if (state.status.type === 'connected') return '#4ade80';
    if (state.status.type === 'error') return '#f87171';
    return '#94a3b8';
  }
  
  function getStatusText(): string {
    if (state.mode === 'local') return 'Local';
    if (state.status.type === 'connected') return 'Connected';
    if (state.status.type === 'error') return 'Error';
    return 'Disconnected';
  }
</script>

<div class="connection-status" class:compact>
  <button 
    class="status-button"
    on:click={() => showDropdown = !showDropdown}
    title="Connection Settings"
  >
    <span class="status-dot" style="background-color: {getStatusColor()}"></span>
    {#if !compact}
      <span class="status-text">{getStatusText()}</span>
      <span class="mode-badge">{state.mode === 'remote' ? 'Jetson' : 'Local'}</span>
    {/if}
    <span class="dropdown-arrow">{showDropdown ? '▲' : '▼'}</span>
  </button>
  
  {#if showDropdown}
    <div class="dropdown">
      <div class="dropdown-header">Connection Mode</div>
      
      <div class="mode-selector">
        <button 
          class="mode-btn" 
          class:active={state.mode === 'local'}
          on:click={() => connectionStore.setMode('local')}
        >
          💻 Local
        </button>
        <button 
          class="mode-btn"
          class:active={state.mode === 'remote'}
          on:click={() => connectionStore.setMode('remote')}
        >
          🖥️ Jetson
        </button>
      </div>
      
      {#if state.mode === 'remote'}
        <div class="remote-settings">
          <label class="url-label">
            Jetson URL
            <input 
              type="text" 
              bind:value={urlInput}
              placeholder="http://10.42.0.10:8080"
              class="url-input"
            />
          </label>
          
          <div class="button-row">
            {#if state.status.type === 'connected'}
              <button class="action-btn disconnect" on:click={handleDisconnect}>
                Disconnect
              </button>
            {:else}
              <button 
                class="action-btn connect" 
                on:click={handleConnect}
                disabled={state.isConnecting}
              >
                {state.isConnecting ? 'Connecting...' : 'Connect'}
              </button>
            {/if}
            
            <button class="action-btn test" on:click={handleTest}>
              Test
            </button>
          </div>
          
          {#if state.status.type === 'connected'}
            <div class="connection-info">
              <div class="info-item">
                <span class="info-label">Version:</span>
                <span class="info-value">{state.status.version}</span>
              </div>
              <div class="info-item">
                <span class="info-label">Uptime:</span>
                <span class="info-value">{Math.floor(state.status.uptime_seconds / 60)}m</span>
              </div>
            </div>
          {:else if state.status.type === 'error'}
            <div class="error-message">
              {state.status.message}
            </div>
          {/if}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .connection-status {
    position: relative;
  }
  
  .status-button {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: white;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .status-button:hover {
    background: rgba(255, 255, 255, 0.15);
  }
  
  .compact .status-button {
    padding: 4px 8px;
  }
  
  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  
  .status-text {
    font-weight: 500;
  }
  
  .mode-badge {
    font-size: 10px;
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .dropdown-arrow {
    font-size: 10px;
    opacity: 0.5;
  }
  
  .dropdown {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    width: 280px;
    background: #1e1e2e;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    z-index: 1000;
    overflow: hidden;
  }
  
  .dropdown-header {
    padding: 12px 16px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: rgba(255, 255, 255, 0.5);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .mode-selector {
    display: flex;
    padding: 12px;
    gap: 8px;
  }
  
  .mode-btn {
    flex: 1;
    padding: 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: white;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .mode-btn:hover {
    background: rgba(255, 255, 255, 0.1);
  }
  
  .mode-btn.active {
    background: rgba(99, 102, 241, 0.2);
    border-color: rgba(99, 102, 241, 0.5);
  }
  
  .remote-settings {
    padding: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
  }
  
  .url-label {
    display: block;
    font-size: 11px;
    color: rgba(255, 255, 255, 0.5);
    margin-bottom: 4px;
  }
  
  .url-input {
    width: 100%;
    padding: 8px 12px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: white;
    font-size: 13px;
    margin-bottom: 12px;
  }
  
  .url-input:focus {
    outline: none;
    border-color: rgba(99, 102, 241, 0.5);
  }
  
  .button-row {
    display: flex;
    gap: 8px;
  }
  
  .action-btn {
    flex: 1;
    padding: 8px 12px;
    border: none;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .action-btn.connect {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
  }
  
  .action-btn.connect:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
  }
  
  .action-btn.connect:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }
  
  .action-btn.disconnect {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
  }
  
  .action-btn.test {
    background: rgba(255, 255, 255, 0.1);
    color: white;
  }
  
  .action-btn.test:hover {
    background: rgba(255, 255, 255, 0.15);
  }
  
  .connection-info {
    margin-top: 12px;
    padding: 8px;
    background: rgba(74, 222, 128, 0.1);
    border-radius: 6px;
    border: 1px solid rgba(74, 222, 128, 0.2);
  }
  
  .info-item {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    padding: 4px 0;
  }
  
  .info-label {
    color: rgba(255, 255, 255, 0.5);
  }
  
  .info-value {
    color: #4ade80;
  }
  
  .error-message {
    margin-top: 12px;
    padding: 8px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 6px;
    color: #f87171;
    font-size: 12px;
  }
</style>
