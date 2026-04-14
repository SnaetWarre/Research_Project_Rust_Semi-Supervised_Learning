import { writable, derived } from 'svelte/store';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { addActivity } from './app';

// Connection mode type
export type ConnectionMode = 'local' | 'remote';

// Connection status types
export interface ConnectionStatusConnected {
  type: 'connected';
  url: string;
  uptime_seconds: number;
  version: string;
}

export interface ConnectionStatusDisconnected {
  type: 'disconnected';
}

export interface ConnectionStatusError {
  type: 'error';
  message: string;
}

export type ConnectionStatus = 
  | ConnectionStatusConnected 
  | ConnectionStatusDisconnected 
  | ConnectionStatusError;

// Connection info from backend
export interface ConnectionInfo {
  mode: ConnectionMode;
  status: ConnectionStatus;
  jetson_url: string;
}

// Connection state
export interface ConnectionState {
  mode: ConnectionMode;
  status: ConnectionStatus;
  jetsonUrl: string;
  isConnecting: boolean;
}

// Create the store
function createConnectionStore() {
  const { subscribe, set, update } = writable<ConnectionState>({
    mode: 'local',
    status: { type: 'disconnected' },
    jetsonUrl: 'http://10.42.0.10:8080',
    isConnecting: false,
  });

  // Initialize by fetching current state from backend
  async function init() {
    try {
      const info = await invoke<ConnectionInfo>('get_connection_info');
      update(state => ({
        ...state,
        mode: info.mode,
        status: parseStatus(info.status),
        jetsonUrl: info.jetson_url,
      }));
    } catch (e) {
      console.error('Failed to get connection info:', e);
    }
  }

  // Parse status from backend format
  function parseStatus(status: any): ConnectionStatus {
    if (typeof status === 'string') {
      if (status === 'disconnected') return { type: 'disconnected' };
    }
    if (status && typeof status === 'object') {
      if ('Connected' in status) {
        return {
          type: 'connected',
          url: status.Connected.url,
          uptime_seconds: status.Connected.uptime_seconds,
          version: status.Connected.version,
        };
      }
      if ('Error' in status) {
        return { type: 'error', message: status.Error };
      }
    }
    return { type: 'disconnected' };
  }

  // Connect to Jetson
  async function connect(url?: string) {
    update(state => ({ ...state, isConnecting: true }));
    
    try {
      const status = await invoke<any>('connect_to_jetson', { url });
      const parsedStatus = parseStatus(status);
      
      update(state => ({
        ...state,
        mode: 'remote',
        status: parsedStatus,
        jetsonUrl: url || state.jetsonUrl,
        isConnecting: false,
      }));

      if (parsedStatus.type === 'connected') {
        addActivity('success', `Connected to Jetson at ${url || 'default URL'}`);
      } else if (parsedStatus.type === 'error') {
        addActivity('error', `Failed to connect: ${parsedStatus.message}`);
      }
      
      return parsedStatus;
    } catch (e) {
      const errorMsg = String(e);
      update(state => ({
        ...state,
        status: { type: 'error', message: errorMsg },
        isConnecting: false,
      }));
      addActivity('error', `Connection failed: ${errorMsg}`);
      throw e;
    }
  }

  // Disconnect from Jetson
  async function disconnect() {
    try {
      await invoke('disconnect_from_jetson');
      update(state => ({
        ...state,
        mode: 'local',
        status: { type: 'disconnected' },
      }));
      addActivity('info', 'Disconnected from Jetson, using local mode');
    } catch (e) {
      console.error('Failed to disconnect:', e);
    }
  }

  // Test connection
  async function testConnection() {
    update(state => ({ ...state, isConnecting: true }));
    
    try {
      const status = await invoke<any>('test_jetson_connection');
      const parsedStatus = parseStatus(status);
      
      update(state => ({
        ...state,
        status: parsedStatus,
        isConnecting: false,
      }));
      
      return parsedStatus;
    } catch (e) {
      update(state => ({
        ...state,
        status: { type: 'error', message: String(e) },
        isConnecting: false,
      }));
      throw e;
    }
  }

  // Set mode
  async function setMode(mode: ConnectionMode) {
    try {
      await invoke('set_connection_mode', { mode });
      update(state => ({ ...state, mode }));
    } catch (e) {
      console.error('Failed to set mode:', e);
    }
  }

  // Listen for connection changes from backend
  listen('connection:changed', (event: any) => {
    const info = event.payload as ConnectionInfo;
    update(state => ({
      ...state,
      mode: info.mode,
      status: parseStatus(info.status),
      jetsonUrl: info.jetson_url,
    }));
  });

  return {
    subscribe,
    init,
    connect,
    disconnect,
    testConnection,
    setMode,
  };
}

export const connectionStore = createConnectionStore();

// Derived store for easy status checks
export const isRemote = derived(
  connectionStore,
  $conn => $conn.mode === 'remote'
);

export const isConnected = derived(
  connectionStore,
  $conn => $conn.mode === 'remote' && $conn.status.type === 'connected'
);

// Remote training state
export interface RemoteTrainingState {
  running: boolean;
  runId: string | null;
  status: 'idle' | 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  output: OutputLine[];
  startedAt: string | null;
}

export interface OutputLine {
  timestamp: string;
  stream: 'stdout' | 'stderr';
  content: string;
}

function createRemoteTrainingStore() {
  const { subscribe, set, update } = writable<RemoteTrainingState>({
    running: false,
    runId: null,
    status: 'idle',
    output: [],
    startedAt: null,
  });

  // Start remote training
  async function start(params: any) {
    try {
      const result = await invoke<any>('start_remote_training', { params });
      update(state => ({
        ...state,
        running: true,
        runId: result.id,
        status: 'running',
        output: [],
        startedAt: new Date().toISOString(),
      }));
      addActivity('success', `Remote training started: ${result.id}`);
      return result;
    } catch (e) {
      addActivity('error', `Failed to start remote training: ${e}`);
      throw e;
    }
  }

  // Stop remote training
  async function stop() {
    try {
      await invoke('stop_remote_training');
      update(state => ({
        ...state,
        running: false,
        status: 'cancelled',
      }));
      addActivity('info', 'Remote training stopped');
    } catch (e) {
      addActivity('error', `Failed to stop training: ${e}`);
      throw e;
    }
  }

  // Fetch current status
  async function fetchStatus() {
    try {
      const status = await invoke<any>('get_remote_training_status');
      update(state => ({
        ...state,
        running: status.running,
        runId: status.current_run?.id || null,
        status: status.current_run?.status || 'idle',
      }));
      return status;
    } catch (e) {
      console.error('Failed to fetch remote training status:', e);
    }
  }

  // Add output line
  function addOutput(line: OutputLine) {
    update(state => ({
      ...state,
      output: [...state.output, line].slice(-500), // Keep last 500 lines
    }));
  }

  // Clear output
  function clearOutput() {
    update(state => ({ ...state, output: [] }));
  }

  return {
    subscribe,
    start,
    stop,
    fetchStatus,
    addOutput,
    clearOutput,
  };
}

export const remoteTrainingStore = createRemoteTrainingStore();
