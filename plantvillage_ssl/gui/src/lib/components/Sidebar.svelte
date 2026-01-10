<script lang="ts">
    import { currentPage, modelInfo } from "$lib/stores/app";
    import {
        LayoutDashboard,
        GraduationCap,
        ScanLine,
        Tags,
        PlayCircle,
        Gauge,
        Leaf,
        TrendingUp,
        FlaskConical,
        Activity,
    } from "lucide-svelte";

    const navItems = [
        { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
        { id: "experiment", label: "Experiments", icon: FlaskConical },
        { id: "inference", label: "Inference", icon: ScanLine },
        { id: "diagnostics", label: "Diagnostics", icon: Activity },
    ];

    function navigate(pageId: string) {
        currentPage.set(pageId);
    }
</script>

<aside class="sidebar">
    <!-- Logo/Header -->
    <div class="sidebar-header">
        <div class="logo-icon">
            <Leaf class="w-5 h-5" style="color: #2142f1;" />
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

    <!-- Status Footer -->
    <div class="sidebar-footer">
        <div class="status-card">
            <div class="status-header">
                <span class="status-label">Model</span>
                <span class="status-dot {$modelInfo.loaded ? 'status-dot-active' : ''}"></span>
            </div>
            <p class="status-text">
                {#if $modelInfo.loaded}
                    {$modelInfo.path?.split("/").pop() || "Loaded"}
                {:else}
                    Not loaded
                {/if}
            </p>
        </div>
    </div>
</aside>

<style>
    .sidebar {
        width: 240px;
        height: 100%;
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        display: flex;
        flex-direction: column;
    }

    .sidebar-header {
        padding: 20px;
        border-bottom: 1px solid #e5e7eb;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .logo-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background-color: #eef2ff;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .logo-title {
        font-size: 16px;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }

    .logo-subtitle {
        font-size: 12px;
        color: #6b7280;
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
        border: none;
        background: transparent;
        color: #4b5563;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
        text-align: left;
    }

    .nav-item:hover {
        background-color: #f3f4f6;
        color: #111827;
    }

    .nav-item-active {
        background-color: #eef2ff;
        color: #2142f1;
    }

    .sidebar-footer {
        padding: 16px;
        border-top: 1px solid #e5e7eb;
    }

    .status-card {
        background-color: #f9fafb;
        border-radius: 8px;
        padding: 12px;
    }

    .status-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }

    .status-label {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #d1d5db;
    }

    .status-dot-active {
        background-color: #10b981;
    }

    .status-text {
        font-size: 13px;
        color: #111827;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
