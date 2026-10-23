//! SSL + incremental learning experiment.
//!
//! Runs the combined pipeline from `training::ssl_incremental` and writes
//! its results and charts.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use colored::Colorize;

use plantvillage_ssl::PlantVillageDataset;

use super::Backend;

/// Run SSL + Incremental Learning combined experiment
pub(crate) fn run_ssl_incremental(
    data_dir: &str,
    output_dir: &str,
    base_classes: usize,
    labeled_samples: usize,
    confidence_threshold: f64,
    seed: u64,
) -> Result<()> {
    use plantvillage_ssl::training::ssl_incremental::{
        generate_ssl_incremental_conclusions, run_ssl_incremental_experiment, SSLIncrementalConfig,
    };

    println!(
        "{}",
        "EXPERIMENT 3: SSL + Incremental Learning".yellow().bold()
    );
    println!("Question: Can pseudo-labeling reduce labeled samples needed for new classes?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Group samples by class
    let mut samples_by_class: HashMap<usize, Vec<(PathBuf, usize)>> = HashMap::new();
    for sample in &dataset.samples {
        samples_by_class
            .entry(sample.label)
            .or_default()
            .push((sample.path.clone(), sample.label));
    }

    let config = SSLIncrementalConfig {
        base_classes,
        labeled_samples_per_new_class: labeled_samples,
        confidence_threshold,
        max_pseudo_labels_per_class: 100,
        base_epochs: 30,
        incremental_epochs: 20,
        batch_size: 64,
        learning_rate: 0.0001,
        seed,
        use_distillation: true,
        distillation_temperature: 2.0,
        distillation_lambda: 1.0,
    };

    println!("{}", "Configuration:".cyan());
    println!("  Base classes: {}", config.base_classes);
    println!(
        "  Labeled samples per new class: {}",
        config.labeled_samples_per_new_class
    );
    println!("  Confidence threshold: {}", config.confidence_threshold);
    println!();

    println!("{}", "Running SSL+IL experiment...".yellow().bold());
    let results = run_ssl_incremental_experiment::<Backend>(samples_by_class, config)?;

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_ssl_incremental_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG charts
    generate_ssl_incremental_chart(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Generate SVG charts for SSL+IL experiment results
fn generate_ssl_incremental_chart(
    results: &plantvillage_ssl::training::ssl_incremental::SSLIncrementalResults,
    output_dir: &str,
) -> Result<()> {
    use plantvillage_ssl::utils::charts::{generate_comparison_chart, BarData};

    // Comparison chart: Without SSL vs With SSL
    let groups = vec![
        (
            "Without SSL",
            vec![
                (
                    "Old Classes",
                    results.without_ssl.old_class_accuracy,
                    "#3498db",
                ),
                (
                    "New Class",
                    results.without_ssl.new_class_accuracy,
                    "#e74c3c",
                ),
                ("Overall", results.without_ssl.overall_accuracy, "#9b59b6"),
            ],
        ),
        (
            "With SSL",
            vec![
                (
                    "Old Classes",
                    results.with_ssl.old_class_accuracy,
                    "#3498db",
                ),
                ("New Class", results.with_ssl.new_class_accuracy, "#2ecc71"),
                ("Overall", results.with_ssl.overall_accuracy, "#9b59b6"),
            ],
        ),
    ];

    let comparison_path = Path::new(output_dir).join("ssl_incremental_comparison.svg");
    generate_comparison_chart(
        "SSL Impact: Incremental Learning Accuracy",
        &groups,
        &comparison_path,
    )?;
    println!("Comparison chart saved to: {:?}", comparison_path);

    // Improvement bar chart
    let improvement_bars = vec![
        BarData {
            label: "Old Class".to_string(),
            value: results.with_ssl.old_class_accuracy - results.without_ssl.old_class_accuracy,
            color: if results.with_ssl.old_class_accuracy >= results.without_ssl.old_class_accuracy
            {
                "#2ecc71".to_string()
            } else {
                "#e74c3c".to_string()
            },
        },
        BarData {
            label: "New Class".to_string(),
            value: results.ssl_improvement,
            color: if results.ssl_improvement >= 0.0 {
                "#2ecc71".to_string()
            } else {
                "#e74c3c".to_string()
            },
        },
        BarData {
            label: "Overall".to_string(),
            value: results.with_ssl.overall_accuracy - results.without_ssl.overall_accuracy,
            color: if results.with_ssl.overall_accuracy >= results.without_ssl.overall_accuracy {
                "#2ecc71".to_string()
            } else {
                "#e74c3c".to_string()
            },
        },
    ];

    // Generate improvement chart
    let improvement_path = Path::new(output_dir).join("ssl_improvement.svg");
    generate_improvement_chart(
        "SSL Improvement over Baseline (% points)",
        &improvement_bars,
        &improvement_path,
    )?;
    println!("Improvement chart saved to: {:?}", improvement_path);

    Ok(())
}

/// Generate a bar chart for showing improvements (can handle any positive values)
fn generate_improvement_chart(
    title: &str,
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

    let y_min = bars.iter().map(|b| b.value).fold(0.0f64, f64::min).min(0.0);
    let y_max = bars
        .iter()
        .map(|b| b.value)
        .fold(0.0f64, f64::max)
        .max(10.0);
    let y_range = (y_max - y_min).max(1.0);
    let y_min_padded = y_min - y_range * 0.1;
    let y_max_padded = y_max + y_range * 0.1;

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

    let zero_y = MARGIN_TOP + plot_height
        - ((0.0 - y_min_padded) / (y_max_padded - y_min_padded)) * plot_height;

    // Grid lines
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = y_min_padded + (i as f64 / 5.0) * (y_max_padded - y_min_padded);

        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT,
            y,
            MARGIN_LEFT + plot_width,
            y,
            COLOR_GRID
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:+.1}%</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Zero line
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT,
        zero_y,
        MARGIN_LEFT + plot_width,
        zero_y,
        COLOR_AXIS
    ));

    // Bars
    for (i, bar) in bars.iter().enumerate() {
        let x = MARGIN_LEFT + (i as f64 * (bar_width + bar_gap)) + bar_gap / 2.0;
        let bar_height = (bar.value.abs() / (y_max_padded - y_min_padded)) * plot_height;

        let y = if bar.value >= 0.0 {
            zero_y - bar_height
        } else {
            zero_y
        };

        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            x,
            y,
            bar_width,
            bar_height.max(2.0),
            bar.color
        ));

        let label_y = if bar.value >= 0.0 {
            y - 8.0
        } else {
            y + bar_height + 18.0
        };
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{}">{:+.1}%</text>"#,
            x + bar_width / 2.0, label_y, COLOR_TEXT, bar.value
        ));

        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{}">{}</text>"#,
            x + bar_width / 2.0, CHART_HEIGHT - 25.0, COLOR_TEXT, bar.label
        ));
    }

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}
