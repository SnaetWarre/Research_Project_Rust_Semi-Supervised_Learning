import { invoke } from '@tauri-apps/api/core';

/**
 * Get the path to the bundled model
 */
export async function getBundledModelPath(): Promise<string> {
    return await invoke<string>('get_bundled_model_path');
}

/**
 * Get the path to the bundled training data
 */
export async function getBundledDataPath(): Promise<string> {
    return await invoke<string>('get_bundled_data_path');
}

/**
 * List all bundled resources
 */
export async function listBundledResources(): Promise<string[]> {
    return await invoke<string[]>('list_bundled_resources');
}

/**
 * Check if we're running on mobile
 */
export function isMobile(): boolean {
    return window.innerWidth < 768;
}

/**
 * Check if bundled resources are available
 */
export async function hasBundledResources(): Promise<boolean> {
    try {
        await getBundledModelPath();
        await getBundledDataPath();
        return true;
    } catch {
        return false;
    }
}
