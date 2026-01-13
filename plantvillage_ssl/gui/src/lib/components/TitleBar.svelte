<script lang="ts">
    import { getCurrentWindow } from "@tauri-apps/api/window";
    import { Minus, Square, X, Maximize2 } from "lucide-svelte";

    const appWindow = getCurrentWindow();

    function minimize() {
        appWindow.minimize();
    }

    async function toggleMaximize() {
        const isMax = await appWindow.isMaximized();
        if (isMax) {
            appWindow.unmaximize();
        } else {
            appWindow.maximize();
        }
    }

    function close() {
        appWindow.close();
    }
</script>

<div class="titlebar" data-tauri-drag-region>
    <div class="title-content">
        <img src="/icons/32x32.png" alt="Logo" class="app-icon" />
        <span class="app-title">PlantVillage SSL Research</span>
    </div>
    <div class="window-controls">
        <button class="control-btn minimize" onclick={minimize}>
            <Minus size={16} />
        </button>
        <button class="control-btn maximize" onclick={toggleMaximize}>
            <Square size={14} />
        </button>
        <button class="control-btn close" onclick={close}>
            <X size={16} />
        </button>
    </div>
</div>

<style>
    .titlebar {
        height: 32px;
        background: var(--bg-panel);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--border-base);
        user-select: none;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
    }

    .title-content {
        display: flex;
        align-items: center;
        padding-left: 12px;
        pointer-events: none; /* Let clicks pass through to drag region */
        gap: 8px;
    }

    .app-icon {
        width: 16px;
        height: 16px;
    }

    .app-title {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-secondary);
        font-family: system-ui, -apple-system, sans-serif;
    }

    .window-controls {
        display: flex;
        height: 100%;
    }

    .control-btn {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 46px;
        height: 100%;
        background: transparent;
        border: none;
        color: var(--text-secondary);
        cursor: default;
        transition: background-color 0.1s;
    }

    .control-btn:hover {
        background-color: var(--bg-hover);
        color: var(--text-main);
    }

    .control-btn.close:hover {
        background-color: var(--error);
        color: white;
    }
</style>
