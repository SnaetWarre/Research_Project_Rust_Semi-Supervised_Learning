<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { open } from "@tauri-apps/plugin-dialog";
    import Card from "$lib/components/Card.svelte";
    import {
        modelInfo,
        datasetInfo,
        trainingState,
        activityLog,
        addActivity,
        diagnosticsState,
    } from "$lib/stores/app";
    import {
        Database,
        Brain,
        Activity,
        Upload,
        CheckCircle,
        AlertCircle,
        Info,
        FlaskConical,
    } from "lucide-svelte";

    let isLoadingModel = $state(false);
    let isLoadingDataset = $state(false);

    async function loadDataset() {
        const selected = await open({
            title: "Select PlantVillage Dataset Directory",
            directory: true,
            multiple: false,
        });

        if (selected) {
            isLoadingDataset = true;
            try {
                const dataDir =
                    typeof selected === "string" ? selected : selected;

                const result = await invoke<{
                    path: string;
                    total_samples: number;
                    num_classes: number;
                    class_names: string[];
                    class_counts: number[];
                }>("get_dataset_stats", { dataDir });

                datasetInfo.set({
                    path: result.path,
                    totalSamples: result.total_samples,
                    numClasses: result.num_classes,
                    classNames: result.class_names,
                    classCounts: result.class_counts,
                });
                addActivity(
                    "success",
                    `Dataset loaded: ${result.total_samples} samples, ${result.num_classes} classes`,
                );
            } catch (e) {
                addActivity("error", `Failed to load dataset: ${e}`);
            } finally {
                isLoadingDataset = false;
            }
        }
    }

    async function loadModel() {
        const selected = await open({
            title: "Select Model File",
            filters: [{ name: "Model", extensions: ["mpk"] }],
        });

        if (selected) {
            isLoadingModel = true;
            try {
                const modelPath =
                    typeof selected === "string"
                        ? selected.replace(/\.mpk$/, "")
                        : selected;

                const result = await invoke<{
                    loaded: boolean;
                    path: string | null;
                    num_classes: number;
                    input_size: number;
                }>("load_model", { modelPath });

                modelInfo.set({
                    loaded: result.loaded,
                    path: result.path,
                    numClasses: result.num_classes,
                    inputSize: result.input_size,
                });
                
                // Clear previous diagnostics when loading a new model
                diagnosticsState.update(state => ({
                    ...state,
                    result: null,
                    lastRunAt: null,
                }));
                
                addActivity("success", `Model loaded successfully`);
            } catch (e) {
                addActivity("error", `Failed to load model: ${e}`);
            } finally {
                isLoadingModel = false;
            }
        }
    }

    function getActivityIcon(type: string) {
        switch (type) {
            case "success":
                return CheckCircle;
            case "warning":
                return AlertCircle;
            case "error":
                return AlertCircle;
            default:
                return Info;
        }
    }

    function getActivityColor(type: string) {
        switch (type) {
            case "success":
                return "color: #10b981;";
            case "warning":
                return "color: #f59e0b;";
            case "error":
                return "color: #ef4444;";
            default:
                return "color: #3b82f6;";
        }
    }
</script>

<div class="dashboard">
    <div class="dashboard-header">
        <div>
            <h1 class="page-title">Dashboard</h1>
            <p class="page-subtitle">Overview of your PlantVillage SSL project</p>
        </div>
        <div class="header-actions">
            <button
                class="btn-secondary"
                onclick={loadDataset}
                disabled={isLoadingDataset}
            >
                <Database style="width: 16px; height: 16px;" />
                {isLoadingDataset ? "Loading..." : "Load Dataset"}
            </button>
            <button
                class="btn-primary"
                onclick={loadModel}
                disabled={isLoadingModel}
            >
                <Upload style="width: 16px; height: 16px;" />
                {isLoadingModel ? "Loading..." : "Load Model"}
            </button>
        </div>
    </div>

    <!-- Stats Grid -->
    <div class="stats-grid">
        <!-- Dataset Card -->
        <div class="stat-card">
            <div class="stat-icon stat-icon-blue">
                <Database style="width: 24px; height: 24px;" />
            </div>
            <div class="stat-content">
                <p class="stat-label">Dataset</p>
                <p class="stat-value">
                    {$datasetInfo
                        ? $datasetInfo.totalSamples.toLocaleString()
                        : "—"}
                </p>
                <p class="stat-desc">
                    {$datasetInfo
                        ? `${$datasetInfo.numClasses} classes`
                        : "Not loaded"}
                </p>
            </div>
        </div>

        <!-- Model Card -->
        <div class="stat-card">
            <div class="stat-icon {$modelInfo.loaded ? 'stat-icon-green' : 'stat-icon-gray'}">
                <Brain style="width: 24px; height: 24px;" />
            </div>
            <div class="stat-content">
                <p class="stat-label">Model</p>
                <p class="stat-value">
                    {$modelInfo.loaded ? "Ready" : "Not loaded"}
                </p>
                <p class="stat-desc">
                    {$modelInfo.loaded
                        ? `${$modelInfo.numClasses} classes`
                        : "Load a model to start"}
                </p>
            </div>
        </div>

        <!-- Experiments Card -->
        <div class="stat-card">
            <div class="stat-icon stat-icon-primary">
                <FlaskConical style="width: 24px; height: 24px;" />
            </div>
            <div class="stat-content">
                <p class="stat-label">Experiments</p>
                <p class="stat-value">3</p>
                <p class="stat-desc">Results available</p>
            </div>
        </div>
    </div>

    <div class="content-grid">
        <!-- Quick Stats -->
        {#if $datasetInfo}
            <Card title="Class Distribution (Top 10)">
                <div class="class-list">
                    {#each $datasetInfo.classNames.slice(0, 10) as className, i}
                        <div class="class-item">
                            <span class="class-name" title={className}>
                                {className.replace(/_/g, " ")}
                            </span>
                            <div class="class-bar-container">
                                <div
                                    class="class-bar"
                                    style="width: {($datasetInfo.classCounts[i] /
                                        Math.max(...$datasetInfo.classCounts)) *
                                        100}%"
                                ></div>
                            </div>
                            <span class="class-count">
                                {$datasetInfo.classCounts[i]}
                            </span>
                        </div>
                    {/each}
                </div>
            </Card>
        {/if}

        <!-- Activity Feed -->
        <Card title="Recent Activity">
            <div class="activity-list">
                {#if $activityLog.length === 0}
                    <p class="no-activity">No recent activity</p>
                {:else}
                    {#each $activityLog as item}
                        <div class="activity-item">
                            <svelte:component
                                this={getActivityIcon(item.type)}
                                style="width: 18px; height: 18px; flex-shrink: 0; {getActivityColor(item.type)}"
                            />
                            <div class="activity-content">
                                <p class="activity-message">{item.message}</p>
                                <p class="activity-time">
                                    {item.timestamp.toLocaleTimeString()}
                                </p>
                            </div>
                        </div>
                    {/each}
                {/if}
            </div>
        </Card>
    </div>
</div>

<style>
    .dashboard {
        max-width: 1200px;
        margin: 0 auto;
    }

    .dashboard-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 24px;
    }

    .page-title {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-main);
        margin: 0 0 4px 0;
    }

    .page-subtitle {
        font-size: 14px;
        color: var(--text-secondary);
        margin: 0;
    }

    .header-actions {
        display: flex;
        gap: 12px;
    }

    .btn-primary, .btn-secondary {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.15s ease;
        border: none;
    }

    .btn-primary {
        background-color: var(--c-white);
        color: var(--c-black);
    }

    .btn-primary:hover:not(:disabled) {
        background-color: var(--c-zinc-200);
    }

    .btn-secondary {
        background-color: transparent;
        color: var(--text-main);
        border: 1px solid var(--border-base);
    }

    .btn-secondary:hover:not(:disabled) {
        background-color: var(--bg-hover);
    }

    .btn-primary:disabled, .btn-secondary:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }

    .stat-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-base);
        border-radius: 12px;
        padding: 20px;
        display: flex;
        gap: 16px;
        align-items: flex-start;
    }

    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }

    .stat-icon-blue {
        background-color: var(--c-accent-dim);
        color: var(--c-accent);
    }

    .stat-icon-green {
        background-color: rgba(16, 185, 129, 0.15);
        color: var(--success);
    }

    .stat-icon-gray {
        background-color: var(--c-zinc-800);
        color: var(--c-zinc-400);
    }

    .stat-icon-primary {
        background-color: var(--c-accent-dim);
        color: var(--c-accent);
    }

    .stat-content {
        flex: 1;
    }

    .stat-label {
        font-size: 13px;
        color: var(--text-secondary);
        margin: 0 0 4px 0;
    }

    .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-main);
        margin: 0 0 4px 0;
    }

    .stat-desc {
        font-size: 13px;
        color: var(--text-secondary);
        margin: 0;
    }

    .content-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
    }

    .class-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        max-height: 320px;
        overflow-y: auto;
    }

    .class-item {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .class-name {
        width: 100px;
        font-size: 13px;
        color: var(--text-main);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .class-bar-container {
        flex: 1;
        height: 8px;
        background-color: var(--c-zinc-800);
        border-radius: 4px;
        overflow: hidden;
    }

    .class-bar {
        height: 100%;
        background-color: var(--c-accent);
        border-radius: 4px;
    }

    .class-count {
        width: 40px;
        text-align: right;
        font-size: 13px;
        color: var(--text-secondary);
    }

    .activity-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
        max-height: 320px;
        overflow-y: auto;
    }

    .activity-item {
        display: flex;
        gap: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-base);
    }

    .activity-item:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }

    .activity-content {
        flex: 1;
        min-width: 0;
    }

    .activity-message {
        font-size: 13px;
        color: var(--text-main);
        margin: 0 0 2px 0;
    }

    .activity-time {
        font-size: 11px;
        color: var(--text-secondary);
        margin: 0;
    }

    .no-activity {
        font-size: 13px;
        color: var(--text-secondary);
        margin: 0;
    }

    @media (max-width: 768px) {
        .stats-grid {
            grid-template-columns: 1fr;
        }

        .content-grid {
            grid-template-columns: 1fr;
        }

        .dashboard-header {
            flex-direction: column;
            gap: 16px;
        }
    }
</style>
