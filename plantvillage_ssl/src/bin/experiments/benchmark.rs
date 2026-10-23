//! Inference benchmark experiment.
//!
//! Measures inference latency and throughput and writes a summary and charts.

use std::fs;
use std::path::Path;

use anyhow::Result;
use colored::Colorize;

/// Run inference benchmark
pub(crate) fn run_inference_benchmark(
    model_path: Option<&str>,
    output_dir: &str,
    warmup: usize,
    iterations: usize,
    batch_size: usize,
    image_size: usize,
) -> Result<()> {
    use plantvillage_ssl::backend::{default_device, DefaultBackend};
    use plantvillage_ssl::inference::{run_benchmark, BenchmarkConfig};

    println!("{}", "BENCHMARK: Inference Performance".yellow().bold());
    println!();

    fs::create_dir_all(output_dir)?;

    println!("{}", "Running GPU benchmark...".green());
    println!();

    let config = BenchmarkConfig {
        warmup_iterations: warmup,
        iterations,
        batch_size,
        measure_memory: true,
        verbose: false,
        output_path: Some(Path::new(output_dir).join("benchmark_results.json")),
    };

    let device = default_device();
    let model_path_ref = model_path.map(Path::new);

    let result =
        run_benchmark::<DefaultBackend>(config.clone(), model_path_ref, image_size, &device)?;

    // Save standard results
    let json = serde_json::to_string_pretty(&result)?;
    let results_path = Path::new(output_dir).join("benchmark_results.json");
    fs::write(&results_path, &json)?;
    println!("\nResults saved to: {:?}", results_path);

    // Generate summary for research paper
    let summary = generate_benchmark_summary(&result);
    let summary_path = Path::new(output_dir).join("benchmark_summary.txt");
    fs::write(&summary_path, &summary)?;
    println!("Summary saved to: {:?}", summary_path);

    // Generate chart
    generate_benchmark_chart(&result, output_dir)?;

    println!();
    println!("{}", "BENCHMARK COMPLETE".green().bold());

    Ok(())
}

/// Generate benchmark summary for research paper
fn generate_benchmark_summary(result: &plantvillage_ssl::inference::BenchmarkOutput) -> String {
    let mut summary = String::new();

    summary.push_str("========================================================================\n");
    summary.push_str("INFERENCE BENCHMARK RESULTS\n");
    summary
        .push_str("========================================================================\n\n");

    summary.push_str(&format!("Framework: {}\n", result.framework));
    summary.push_str(&format!("Device: {}\n", result.device));
    summary.push_str(&format!("Timestamp: {}\n", result.timestamp));
    summary.push_str("Platform: CUDA GPU\n");
    summary.push('\n');

    summary.push_str("CONFIGURATION:\n");
    summary.push_str(&format!(
        "  Image Size: {}x{}\n",
        result.image_size, result.image_size
    ));
    summary.push_str(&format!("  Batch Size: {}\n", result.batch_size));
    summary.push_str(&format!(
        "  Warmup Iterations: {}\n",
        result.warmup_iterations
    ));
    summary.push_str(&format!(
        "  Benchmark Iterations: {}\n",
        result.num_iterations
    ));
    summary.push('\n');

    summary.push_str("LATENCY METRICS:\n");
    summary.push_str(&format!("  Mean:   {:8.2} ms\n", result.mean_ms));
    summary.push_str(&format!("  Std:    {:8.2} ms\n", result.std_ms));
    summary.push_str(&format!("  Min:    {:8.2} ms\n", result.min_ms));
    summary.push_str(&format!("  Max:    {:8.2} ms\n", result.max_ms));
    summary.push_str(&format!("  P50:    {:8.2} ms\n", result.p50_ms));
    summary.push_str(&format!("  P95:    {:8.2} ms\n", result.p95_ms));
    summary.push_str(&format!("  P99:    {:8.2} ms\n", result.p99_ms));
    summary.push('\n');

    summary.push_str("THROUGHPUT:\n");
    summary.push_str(&format!("  {:8.1} images/second\n", result.throughput_fps));
    summary.push('\n');

    if result.model_size_mb > 0.0 {
        summary.push_str("MODEL:\n");
        summary.push_str(&format!("  Size: {:.2} MB\n", result.model_size_mb));
        summary.push('\n');
    }

    // Check against targets
    summary.push_str("TARGET COMPLIANCE:\n");
    let target_200ms = result.mean_ms <= 200.0;
    let target_500ms = result.mean_ms <= 500.0;
    summary.push_str(&format!(
        "  200ms target: {}\n",
        if target_200ms { "PASS" } else { "FAIL" }
    ));
    summary.push_str(&format!(
        "  500ms target: {}\n",
        if target_500ms { "PASS" } else { "FAIL" }
    ));

    summary
}

/// Generate benchmark visualization chart
fn generate_benchmark_chart(
    result: &plantvillage_ssl::inference::BenchmarkOutput,
    output_dir: &str,
) -> Result<()> {
    use plantvillage_ssl::utils::charts::BarData;

    let bars = vec![
        BarData {
            label: "Mean".to_string(),
            value: result.mean_ms,
            color: "#3498db".to_string(),
        },
        BarData {
            label: "P50".to_string(),
            value: result.p50_ms,
            color: "#2ecc71".to_string(),
        },
        BarData {
            label: "P95".to_string(),
            value: result.p95_ms,
            color: "#f39c12".to_string(),
        },
        BarData {
            label: "P99".to_string(),
            value: result.p99_ms,
            color: "#e74c3c".to_string(),
        },
    ];

    // We need a custom chart since this isn't percentages
    let chart_path = Path::new(output_dir).join("latency_distribution.svg");
    generate_latency_chart(
        &format!("Inference Latency ({} iterations)", result.num_iterations),
        "Latency (ms)",
        &bars,
        &chart_path,
    )?;
    println!("Latency chart saved to: {:?}", chart_path);

    Ok(())
}

/// Generate a bar chart for latency metrics (in ms, not %)
fn generate_latency_chart(
    title: &str,
    y_label: &str,
    bars: &[plantvillage_ssl::utils::charts::BarData],
    output_path: &Path,
) -> std::io::Result<()> {
    const CHART_WIDTH: f64 = 800.0;
    const CHART_HEIGHT: f64 = 500.0;
    const MARGIN_TOP: f64 = 60.0;
    const MARGIN_RIGHT: f64 = 40.0;
    const MARGIN_BOTTOM: f64 = 80.0;
    const MARGIN_LEFT: f64 = 80.0;
    const COLOR_GRID: &str = "#ecf0f1";
    const COLOR_AXIS: &str = "#2c3e50";
    const COLOR_TEXT: &str = "#2c3e50";

    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    let y_max = bars.iter().map(|b| b.value).fold(0.0f64, f64::max) * 1.2;

    let bar_width = (plot_width / bars.len() as f64) * 0.7;
    let bar_gap = (plot_width / bars.len() as f64) * 0.3;

    let mut svg = String::new();

    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        CHART_WIDTH, CHART_HEIGHT, CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        CHART_WIDTH, CHART_HEIGHT
    ));

    svg.push_str(&format!(
        r#"<text x="{}" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
        CHART_WIDTH / 2.0, COLOR_TEXT, title
    ));

    // Grid lines
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = (i as f64 / 5.0) * y_max;

        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT,
            y,
            MARGIN_LEFT + plot_width,
            y,
            COLOR_GRID
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:.1} ms</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Axes
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT,
        MARGIN_TOP + plot_height,
        MARGIN_LEFT + plot_width,
        MARGIN_TOP + plot_height,
        COLOR_AXIS
    ));

    // Y-axis label
    svg.push_str(&format!(
        r#"<text x="20" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{}" transform="rotate(-90 20 {})">{}</text>"#,
        CHART_HEIGHT / 2.0, COLOR_TEXT, CHART_HEIGHT / 2.0, y_label
    ));

    // Bars
    for (i, bar) in bars.iter().enumerate() {
        let x = MARGIN_LEFT + (i as f64 * (bar_width + bar_gap)) + bar_gap / 2.0;
        let bar_height = (bar.value / y_max) * plot_height;
        let y = MARGIN_TOP + plot_height - bar_height;

        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            x, y, bar_width, bar_height, bar.color
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="{}">{:.1} ms</text>"#,
            x + bar_width / 2.0, y - 8.0, COLOR_TEXT, bar.value
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{}">{}</text>"#,
            x + bar_width / 2.0, MARGIN_TOP + plot_height + 25.0, COLOR_TEXT, bar.label
        ));
    }

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}
