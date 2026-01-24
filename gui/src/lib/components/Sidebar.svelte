<script lang="ts">
    import { currentPage, modelInfo } from "$lib/stores/app";
    import {
        LayoutDashboard,
        ScanLine,
        Leaf,
        FlaskConical,
        Activity,
        Sparkles,
    } from "lucide-svelte";

    interface Props {
        open?: boolean;
        onNavigate?: () => void;
    }

    let { open = $bindable(false), onNavigate }: Props = $props();

    const navItems = [
        { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
        { id: "demo", label: "SSL Demo", icon: Sparkles },
        { id: "experiment", label: "Experiments", icon: FlaskConical },
        { id: "inference", label: "Inference", icon: ScanLine },
        { id: "diagnostics", label: "Diagnostics", icon: Activity },
    ];

    function navigate(pageId: string) {
        currentPage.set(pageId);
        if (onNavigate) {
            onNavigate();
        }
    }
</script>

<aside class="sidebar {open ? 'sidebar-open' : ''}">
    <!-- Logo/Header -->
    <div class="sidebar-header">
        <div class="logo-icon">
            <Leaf class="w-5 h-5" style="color: var(--c-white);" />
        </div>
        <div>
            <h1 class="logo-title">PlantVillage</h1>
            <p class="logo-subtitle">SSL Research</p>
        </div>
    </div>

    <!-- Navigation -->
    <nav class="sidebar-nav">
        <ul class="nav-list">
            {#each navItems as item}
                <li>
                    <button
                        class="nav-item {$currentPage === item.id ? 'nav-item-active' : ''}"
                        onclick={() => navigate(item.id)}
                    >
                        <svelte:component this={item.icon} class="w-5 h-5" />
                        <span>{item.label}</span>
                    </button>
                </li>
            {/each}
        </ul>
    </nav>
</aside>

<style>
    .sidebar {
        width: 240px;
        height: 100%;
        background-color: var(--bg-panel);
        border-right: 1px solid var(--border-base);
        display: none; /* Hidden by default on mobile */
        flex-direction: column;
    }

    @media (min-width: 769px) {
        .sidebar {
            display: flex; /* Only show on desktop */
        }
    }

    .sidebar-header {
        padding: 20px;
        border-bottom: 1px solid var(--border-base);
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .logo-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background-color: var(--c-zinc-800);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .logo-title {
        font-size: 16px;
        font-weight: 700;
        color: var(--text-main);
        margin: 0;
    }

    .logo-subtitle {
        font-size: 12px;
        color: var(--text-secondary);
        margin: 0;
    }

    .sidebar-nav {
        flex: 1;
        padding: 16px 12px;
        overflow-y: auto;
    }

    .nav-list {
        list-style: none;
        margin: 0;
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .nav-item {
        width: 100%;
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 12px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--text-secondary);
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
        text-align: left;
    }

    .nav-item:hover {
        background-color: var(--bg-hover);
        color: var(--text-main);
    }

    .nav-item-active {
        background-color: var(--bg-hover);
        border-color: var(--border-highlight);
        color: var(--text-main);
    }
</style>
