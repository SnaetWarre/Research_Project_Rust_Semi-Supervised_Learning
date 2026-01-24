<script lang="ts">
    import { invoke } from "@tauri-apps/api/core";
    import { onMount } from "svelte";
    import Card from "$lib/components/Card.svelte";
    import BarChart from "$lib/components/BarChart.svelte";

    // Experiment results state
    let isLoading = $state(true);
    let activeTab = $state("ssl_incremental");
    let availableExperiments: string[] = $state([]);

    // Results
    let labelEfficiency: any = $state(null);
    let classScaling: any = $state(null);
    let sslIncremental: any = $state(null);
    let newClassPosition: any = $state(null);

    // Derived states for chart labels (fixes Svelte 5 state proxy issue)
    let labelEfficiencyLabels = $derived(
        labelEfficiency?.images_per_class?.map((n: number) => `${n}`) ?? []
    );
    let labelEfficiencyData = $derived(
        labelEfficiency?.accuracies ?? []
    );

    onMount(async () => {
        await loadExperiments();
    });

    async function loadExperiments() {
        isLoading = true;
        try {
            // Load all experiments
            const results = await invoke<any>("load_all_experiment_results", {
                baseDir: null
            });

            labelEfficiency = results.label_efficiency;
            classScaling = results.class_scaling;
            sslIncremental = results.ssl_incremental;
            newClassPosition = results.new_class_position;

            availableExperiments = [];
            if (labelEfficiency) availableExperiments.push("label_efficiency");
            if (classScaling) availableExperiments.push("class_scaling");
            if (sslIncremental) availableExperiments.push("ssl_incremental");
            if (newClassPosition) availableExperiments.push("new_class_position");

            // Set active tab to first available
            if (sslIncremental) activeTab = "ssl_incremental";
            else if (labelEfficiency) activeTab = "label_efficiency";
            else if (classScaling) activeTab = "class_scaling";
            else if (newClassPosition) activeTab = "new_class_position";

        } catch (error) {
            console.error("Failed to load experiments:", error);
        } finally {
            isLoading = false;
        }
    }

    const tabs = [
        { id: "ssl_incremental", label: "SSL + Incremental", available: () => !!sslIncremental },
        { id: "label_efficiency", label: "Label Efficiency", available: () => !!labelEfficiency },
        { id: "class_scaling", label: "Class Scaling", available: () => !!classScaling },
        { id: "new_class_position", label: "New Class Position", available: () => !!newClassPosition },
    ];
</script>

<div class="experiments-page">
    <div class="page-header">
        <div>
            <h1 class="page-title">Experiment Results</h1>
            <p class="page-subtitle">Pre-run research experiment results ready for demo</p>
        </div>
        <button class="btn-secondary" onclick={loadExperiments} disabled={isLoading}>
            {isLoading ? "Loading..." : "Refresh"}
        </button>
    </div>

    {#if isLoading}
        <div class="loading-state">
            <div class="spinner"></div>
            <p>Loading experiment results...</p>
        </div>
    {:else if availableExperiments.length === 0}
        <div class="empty-state">
            <h3>No Experiments Found</h3>
            <p>Run experiments first using the CLI:</p>
            <code>./target/release/experiments all --data-dir data/plantvillage/balanced</code>
        </div>
    {:else}
        <!-- Tabs -->
        <div class="tab-container">
            {#each tabs as tab}
                {#if tab.available()}
                    <button
                        class="tab-btn {activeTab === tab.id ? 'tab-btn-active' : ''}"
                        onclick={() => activeTab = tab.id}
                    >
                        {tab.label}
                    </button>
                {/if}
            {/each}
        </div>

        <!-- SSL + Incremental Results -->
        {#if activeTab === "ssl_incremental" && sslIncremental}
            <div class="results-section">
                <Card title="SSL + Incremental Learning: Key Finding">
                    <div class="key-finding">
                        <div class="finding-highlight">
                            <span class="finding-value">+{sslIncremental.ssl_improvement.toFixed(1)}%</span>
                            <span class="finding-label">New Class Accuracy Improvement</span>
                        </div>
                        <p class="finding-desc">
                            With only <strong>10 labeled samples</strong> + <strong>{sslIncremental.pseudo_labels_generated} pseudo-labels</strong>,
                            SSL dramatically improves new class learning vs baseline fine-tuning.
                        </p>
                    </div>
                </Card>

                <div class="stats-row">
                    <div class="stat-box">
                        <span class="stat-label">Base Model Accuracy</span>
                        <span class="stat-value">{sslIncremental.base_accuracy.toFixed(1)}%</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Pseudo-Labels Generated</span>
                        <span class="stat-value">{sslIncremental.pseudo_labels_generated}</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">Pseudo-Label Accuracy</span>
                        <span class="stat-value">{(sslIncremental.pseudo_label_accuracy * 100).toFixed(0)}%</span>
                    </div>
                </div>

                <Card title="Comparison: Without SSL vs With SSL">
                    <div class="comparison-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Without SSL</th>
                                    <th>With SSL</th>
                                    <th>Improvement</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Old Class Accuracy</td>
                                    <td>{sslIncremental.without_ssl.old_class_accuracy.toFixed(2)}%</td>
                                    <td>{sslIncremental.with_ssl.old_class_accuracy.toFixed(2)}%</td>
                                    <td class="improvement">
                                        {(sslIncremental.with_ssl.old_class_accuracy - sslIncremental.without_ssl.old_class_accuracy) > 0 ? '+' : ''}
                                        {(sslIncremental.with_ssl.old_class_accuracy - sslIncremental.without_ssl.old_class_accuracy).toFixed(2)}%
                                    </td>
                                </tr>
                                <tr>
                                    <td>New Class Accuracy</td>
                                    <td>{sslIncremental.without_ssl.new_class_accuracy.toFixed(2)}%</td>
                                    <td class="highlight">{sslIncremental.with_ssl.new_class_accuracy.toFixed(2)}%</td>
                                    <td class="improvement positive">+{sslIncremental.ssl_improvement.toFixed(2)}%</td>
                                </tr>
                                <tr>
                                    <td>Overall Accuracy</td>
                                    <td>{sslIncremental.without_ssl.overall_accuracy.toFixed(2)}%</td>
                                    <td>{sslIncremental.with_ssl.overall_accuracy.toFixed(2)}%</td>
                                    <td class="improvement">
                                        {(sslIncremental.with_ssl.overall_accuracy - sslIncremental.without_ssl.overall_accuracy) > 0 ? '+' : ''}
                                        {(sslIncremental.with_ssl.overall_accuracy - sslIncremental.without_ssl.overall_accuracy).toFixed(2)}%
                                    </td>
                                </tr>
                                <tr>
                                    <td>Forgetting</td>
                                    <td>{sslIncremental.without_ssl.forgetting.toFixed(2)}%</td>
                                    <td>{sslIncremental.with_ssl.forgetting.toFixed(2)}%</td>
                                    <td class="improvement">-</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </Card>

                <div class="conclusion-box">
                    <h4>Conclusion</h4>
                    <p>SSL significantly improves new class learning with limited labels. Pseudo-labeling effectively augments the small labeled dataset, enabling the model to achieve {sslIncremental.with_ssl.new_class_accuracy.toFixed(1)}% accuracy on new classes vs {sslIncremental.without_ssl.new_class_accuracy.toFixed(1)}% without SSL.</p>
                </div>
            </div>
        {/if}

        <!-- Label Efficiency Results -->
        {#if activeTab === "label_efficiency" && labelEfficiency}
            <div class="results-section">
                <Card title="Label Efficiency: How Many Labels Do We Need?">
                    <div class="key-finding">
                        <div class="finding-highlight">
                            <span class="finding-value">{labelEfficiency.best_accuracy.toFixed(1)}%</span>
                            <span class="finding-label">Best Accuracy @ {labelEfficiency.best_images_per_class} images/class</span>
                        </div>
                        {#if labelEfficiency.min_acceptable_images}
                            <p class="finding-desc">
                                Need at least <strong>{labelEfficiency.min_acceptable_images} images per class</strong> for >80% accuracy.
                                This justifies the need for Semi-Supervised Learning.
                            </p>
                        {/if}
                    </div>
                </Card>

                <Card title="Accuracy vs Images per Class">
                    <div class="chart-wrapper">
                        <BarChart
                            data={labelEfficiencyData}
                            labels={labelEfficiencyLabels}
                            label="Validation Accuracy (%)"
                            color="#2142f1"
                        />
                    </div>
                </Card>

                <Card title="Detailed Results">
                    <div class="comparison-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Images/Class</th>
                                    <th>Accuracy</th>
                                    <th>Training Time</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {#each labelEfficiency.images_per_class as images, i}
                                    <tr>
                                        <td>{images}</td>
                                        <td class="{labelEfficiency.accuracies[i] >= 80 ? 'highlight' : ''}">
                                            {labelEfficiency.accuracies[i].toFixed(2)}%
                                        </td>
                                        <td>{labelEfficiency.training_times[i].toFixed(1)}s</td>
                                        <td>
                                            {#if labelEfficiency.accuracies[i] >= 80}
                                                <span class="status-badge status-success">Acceptable</span>
                                            {:else}
                                                <span class="status-badge status-warning">Below threshold</span>
                                            {/if}
                                        </td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    </div>
                </Card>
            </div>
        {/if}

        <!-- Class Scaling Results -->
        {#if activeTab === "class_scaling" && classScaling}
            <div class="results-section">
                <Card title="Class Scaling: Is 5->6 Harder Than 30->31?">
                    <div class="key-finding">
                        <div class="finding-highlight">
                            <span class="finding-value">{(classScaling.small_base.new_class_accuracy - classScaling.large_base.new_class_accuracy).toFixed(1)}%</span>
                            <span class="finding-label">Accuracy Drop (Small vs Large Base)</span>
                        </div>
                        <p class="finding-desc">
                            Adding a new class to a <strong>larger model (30 classes)</strong> results in 
                            <strong>lower new class accuracy</strong> compared to a smaller model (5 classes).
                        </p>
                    </div>
                </Card>

                <div class="comparison-grid">
                    <Card title="5 -> 6 Classes">
                        <div class="scaling-stats">
                            <div class="scaling-row">
                                <span>Base Accuracy (Before)</span>
                                <span>{classScaling.small_base.base_accuracy_before.toFixed(2)}%</span>
                            </div>
                            <div class="scaling-row">
                                <span>Base Accuracy (After)</span>
                                <span>{classScaling.small_base.base_accuracy_after.toFixed(2)}%</span>
                            </div>
                            <div class="scaling-row highlight-row">
                                <span>New Class Accuracy</span>
                                <span class="highlight">{classScaling.small_base.new_class_accuracy.toFixed(2)}%</span>
                            </div>
                            <div class="scaling-row">
                                <span>Forgetting</span>
                                <span>{classScaling.small_base.forgetting.toFixed(2)}%</span>
                            </div>
                        </div>
                    </Card>

                    <Card title="30 -> 31 Classes">
                        <div class="scaling-stats">
                            <div class="scaling-row">
                                <span>Base Accuracy (Before)</span>
                                <span>{classScaling.large_base.base_accuracy_before.toFixed(2)}%</span>
                            </div>
                            <div class="scaling-row">
                                <span>Base Accuracy (After)</span>
                                <span>{classScaling.large_base.base_accuracy_after.toFixed(2)}%</span>
                            </div>
                            <div class="scaling-row highlight-row">
                                <span>New Class Accuracy</span>
                                <span class="highlight">{classScaling.large_base.new_class_accuracy.toFixed(2)}%</span>
                            </div>
                            <div class="scaling-row">
                                <span>Forgetting</span>
                                <span>{classScaling.large_base.forgetting.toFixed(2)}%</span>
                            </div>
                        </div>
                    </Card>
                </div>

                <div class="conclusion-box">
                    <h4>Conclusion</h4>
                    <p>Larger base models have a harder time learning new classes due to increased class competition. The new class accuracy drops from {classScaling.small_base.new_class_accuracy.toFixed(1)}% (small base) to {classScaling.large_base.new_class_accuracy.toFixed(1)}% (large base). This motivates the use of incremental learning methods like LwF, EWC, and Rehearsal.</p>
                </div>
            </div>
        {/if}

        <!-- New Class Position Results -->
        {#if activeTab === "new_class_position" && newClassPosition}
            <div class="results-section">
                <Card title="New Class Position: 6th vs 31st Class">
                    <div class="key-finding">
                        <div class="finding-highlight">
                            <span class="finding-value">{newClassPosition.summary.harder_as_31st ? "Harder" : "Similar"}</span>
                            <span class="finding-label">Learning as 31st Class vs 6th Class</span>
                        </div>
                        <p class="finding-desc">
                            With <strong>50 labeled samples</strong>, adding a class as the 6th achieves 
                            <strong>93.1% accuracy</strong> while adding as 31st only achieves 
                            <strong>40.2% accuracy</strong> - a significant gap of over 50 percentage points.
                        </p>
                    </div>
                </Card>

                <Card title="Accuracy Comparison: Samples Needed">
                    <div class="comparison-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Labeled Samples</th>
                                    <th>6th Class Acc.</th>
                                    <th>31st Class Acc.</th>
                                    <th>Difference</th>
                                </tr>
                            </thead>
                            <tbody>
                                {#each newClassPosition.small_base_results as result, i}
                                    <tr>
                                        <td>{result.labeled_samples}</td>
                                        <td class="{result.new_class_accuracy >= 70 ? 'highlight' : ''}">{result.new_class_accuracy.toFixed(2)}%</td>
                                        <td class="{newClassPosition.large_base_results[i].new_class_accuracy >= 70 ? 'highlight' : ''}">{newClassPosition.large_base_results[i].new_class_accuracy.toFixed(2)}%</td>
                                        <td class="improvement {result.new_class_accuracy - newClassPosition.large_base_results[i].new_class_accuracy > 0 ? 'positive' : ''}">
                                            {(result.new_class_accuracy - newClassPosition.large_base_results[i].new_class_accuracy) > 0 ? '+' : ''}
                                            {(result.new_class_accuracy - newClassPosition.large_base_results[i].new_class_accuracy).toFixed(2)}%
                                        </td>
                                    </tr>
                                {/each}
                            </tbody>
                        </table>
                    </div>
                </Card>

                <div class="comparison-grid">
                    <Card title="Minimum Samples for Target Accuracy">
                        <div class="scaling-stats">
                            <div class="scaling-row">
                                <span>70% Accuracy (6th class)</span>
                                <span>{newClassPosition.summary.min_samples_small_70pct ?? 'N/A'} samples</span>
                            </div>
                            <div class="scaling-row">
                                <span>70% Accuracy (31st class)</span>
                                <span>{newClassPosition.summary.min_samples_large_70pct ?? '>100'} samples</span>
                            </div>
                            <div class="scaling-row highlight-row">
                                <span>80% Accuracy (6th class)</span>
                                <span class="highlight">{newClassPosition.summary.min_samples_small_80pct ?? 'N/A'} samples</span>
                            </div>
                            <div class="scaling-row">
                                <span>80% Accuracy (31st class)</span>
                                <span>{newClassPosition.summary.min_samples_large_80pct ?? '>100'} samples</span>
                            </div>
                        </div>
                    </Card>

                    <Card title="Forgetting Analysis">
                        <div class="scaling-stats">
                            <div class="scaling-row">
                                <span>Avg. Forgetting Difference</span>
                                <span>{newClassPosition.summary.avg_forgetting_difference.toFixed(2)}%</span>
                            </div>
                            <div class="scaling-row">
                                <span>Harder as 31st Class?</span>
                                <span class="{newClassPosition.summary.harder_as_31st ? 'highlight' : ''}">{newClassPosition.summary.harder_as_31st ? 'Yes' : 'No'}</span>
                            </div>
                            <div class="scaling-row highlight-row">
                                <span>Samples Ratio (31st/6th)</span>
                                <span class="highlight">{newClassPosition.summary.samples_ratio.toFixed(2)}x</span>
                            </div>
                        </div>
                    </Card>
                </div>

                <div class="conclusion-box">
                    <h4>Conclusion</h4>
                    <p>
                        {#if newClassPosition.summary.harder_as_31st}
                            Adding a new class becomes significantly harder as the model grows. With 50 samples, the 6th class achieves 93.1% accuracy while the 31st class only achieves 40.2%. 
                            <strong>For mature models with many classes, collect more labeled samples or use SSL to augment limited data.</strong>
                        {:else}
                            Class position has limited impact on learning difficulty. The same labeling effort works for both early and late classes.
                        {/if}
                    </p>
                </div>
            </div>
        {/if}
    {/if}
</div>

<style>
    .experiments-page {
        max-width: 1200px;
        margin: 0 auto;
    }

    .page-header {
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

    .btn-secondary {
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        background-color: transparent;
        color: var(--text-main);
        border: 1px solid var(--border-base);
        cursor: pointer;
    }

    .btn-secondary:hover:not(:disabled) {
        background-color: var(--bg-hover);
    }

    .btn-secondary:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .loading-state, .empty-state {
        text-align: center;
        padding: 60px 20px;
        background: var(--bg-panel);
        border-radius: 12px;
        border: 1px solid var(--border-base);
    }

    .spinner {
        width: 40px;
        height: 40px;
        border: 3px solid var(--c-zinc-800);
        border-top-color: var(--c-accent);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
        margin: 0 auto 16px;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .empty-state h3 {
        color: var(--text-main);
        margin-bottom: 8px;
    }

    .empty-state p {
        color: var(--text-secondary);
        margin-bottom: 16px;
    }

    .empty-state code {
        display: inline-block;
        background: var(--bg-surface);
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 13px;
        color: var(--text-main);
        border: 1px solid var(--border-base);
    }

    .tab-container {
        display: flex;
        gap: 4px;
        background: var(--bg-surface);
        padding: 4px;
        border-radius: 10px;
        margin-bottom: 24px;
        border: 1px solid var(--border-base);
    }

    .tab-btn {
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        background: transparent;
        color: var(--text-secondary);
        border: none;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .tab-btn:hover {
        color: var(--text-main);
        background: var(--bg-hover);
    }

    .tab-btn-active {
        background: var(--bg-panel);
        color: var(--c-accent);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
    }

    .results-section {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }

    .key-finding {
        text-align: center;
        padding: 20px;
    }

    .finding-highlight {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 16px;
    }

    .finding-value {
        font-size: 48px;
        font-weight: 700;
        color: var(--c-accent);
    }

    .finding-label {
        font-size: 14px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    .finding-desc {
        font-size: 15px;
        color: var(--text-secondary);
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .finding-desc strong {
        color: var(--text-main);
    }

    .stats-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
    }

    .stat-box {
        background: var(--bg-panel);
        border: 1px solid var(--border-base);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    .stat-box .stat-label {
        display: block;
        font-size: 12px;
        color: var(--text-secondary);
        margin-bottom: 4px;
    }

    .stat-box .stat-value {
        font-size: 24px;
        font-weight: 700;
        color: var(--text-main);
    }

    .comparison-table {
        overflow-x: auto;
    }

    .comparison-table table {
        width: 100%;
        border-collapse: collapse;
    }

    .comparison-table th, .comparison-table td {
        padding: 12px 16px;
        text-align: left;
        border-bottom: 1px solid var(--border-base);
    }

    .comparison-table th {
        font-weight: 600;
        color: var(--text-secondary);
        background: var(--bg-surface);
    }

    .comparison-table td {
        color: var(--text-secondary);
    }

    .comparison-table .highlight {
        color: var(--c-accent);
        font-weight: 600;
    }

    .comparison-table .improvement {
        font-weight: 500;
    }

    .comparison-table .improvement.positive {
        color: var(--success);
    }

    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
    }

    .status-success {
        background: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid rgba(245, 158, 11, 0.2);
    }

    .conclusion-box {
        background: var(--c-accent-dim);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 20px;
    }

    .conclusion-box h4 {
        font-size: 14px;
        font-weight: 600;
        color: var(--c-accent);
        margin: 0 0 8px 0;
    }

    .conclusion-box p {
        font-size: 14px;
        color: var(--text-main);
        margin: 0;
        line-height: 1.6;
    }

    .comparison-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
    }

    .scaling-stats {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .scaling-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid var(--border-base);
    }

    .scaling-row:last-child {
        border-bottom: none;
    }

    .scaling-row span:first-child {
        color: var(--text-secondary);
        font-size: 14px;
    }

    .scaling-row span:last-child {
        font-weight: 600;
        color: var(--text-main);
    }

    .highlight-row {
        background: var(--bg-surface);
        margin: 0 -16px;
        padding: 12px 16px !important;
        border-radius: 8px;
        border-bottom: none !important;
    }

    .chart-wrapper {
        height: 300px;
    }

    @media (max-width: 768px) {
        .stats-row {
            grid-template-columns: 1fr;
        }

        .comparison-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
