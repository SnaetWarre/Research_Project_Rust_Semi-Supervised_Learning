//! Experiment 4: new class position effect.
//!
//! Adds a 6th or a 31st class with varying numbers of labeled samples and
//! compares new-class accuracy and forgetting.

use std::fs;
use std::path::Path;

use anyhow::Result;
use chrono::Local;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use plantvillage_ssl::PlantVillageDataset;

use super::incremental::run_incremental_experiment;
use super::Backend;

/// Results from new class position experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NewClassPositionResults {
    /// Results for small base (5 classes) with varying labeled samples
    small_base_results: Vec<PositionLabelResult>,
    /// Results for large base (30 classes) with varying labeled samples
    large_base_results: Vec<PositionLabelResult>,
    /// Summary statistics
    summary: PositionSummary,
}

/// Result for a single combination of base size and labeled samples
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionLabelResult {
    /// Number of base classes
    base_classes: usize,
    /// Number of labeled samples for the new class
    labeled_samples: usize,
    /// Accuracy on the new class
    new_class_accuracy: f64,
    /// Accuracy on base classes after adding
    base_accuracy_after: f64,
    /// Forgetting (accuracy drop on old classes)
    forgetting: f64,
    /// Overall accuracy
    overall_accuracy: f64,
    /// Training time in seconds
    training_time: f64,
}

/// Summary of the position experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionSummary {
    /// Minimum samples for 70% accuracy on new class (small base)
    min_samples_small_70pct: Option<usize>,
    /// Minimum samples for 70% accuracy on new class (large base)
    min_samples_large_70pct: Option<usize>,
    /// Minimum samples for 80% accuracy on new class (small base)
    min_samples_small_80pct: Option<usize>,
    /// Minimum samples for 80% accuracy on new class (large base)
    min_samples_large_80pct: Option<usize>,
    /// Average forgetting difference (large - small)
    avg_forgetting_difference: f64,
    /// Is learning harder as 31st class?
    harder_as_31st: bool,
    /// Samples needed ratio (large/small for equivalent accuracy)
    samples_ratio: f64,
}

/// Run new class position experiment: Does adding a class as 6th vs 31st need different amounts of labels?
pub(crate) fn run_new_class_position(
    data_dir: &str,
    output_dir: &str,
    epochs: usize,
    seed: u64,
) -> Result<()> {
    println!(
        "{}",
        "EXPERIMENT 4: New Class Position Effect".yellow().bold()
    );
    println!("Question: Does adding a class as 6th vs 31st require different amounts of labeled samples?");
    println!("Sub-question: How does the number of existing classes affect label efficiency for new classes?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Test different numbers of labeled samples for the new class
    let labeled_samples_tests = vec![5, 10, 25, 50, 100];

    let mut small_base_results: Vec<PositionLabelResult> = Vec::new();
    let mut large_base_results: Vec<PositionLabelResult> = Vec::new();

    // Part A: 5 → 6 classes with varying labeled samples
    println!(
        "{}",
        "Part A: Adding 6th class (5 base classes)".yellow().bold()
    );
    println!();

    for &n_samples in &labeled_samples_tests {
        println!(
            "  Testing with {} labeled samples for new class...",
            n_samples
        );
        let result =
            run_incremental_experiment::<Backend>(&dataset, 5, 1, n_samples, epochs, seed)?;
        println!(
            "    New class: {:.2}%, Base after: {:.2}%, Forgetting: {:.2}%",
            result.new_class_accuracy, result.base_accuracy_after, result.forgetting
        );
        small_base_results.push(PositionLabelResult {
            base_classes: 5,
            labeled_samples: n_samples,
            new_class_accuracy: result.new_class_accuracy,
            base_accuracy_after: result.base_accuracy_after,
            forgetting: result.forgetting,
            overall_accuracy: result.overall_accuracy,
            training_time: result.training_time,
        });
    }
    println!();

    // Part B: 30 → 31 classes with varying labeled samples
    println!(
        "{}",
        "Part B: Adding 31st class (30 base classes)"
            .yellow()
            .bold()
    );
    println!();

    for &n_samples in &labeled_samples_tests {
        println!(
            "  Testing with {} labeled samples for new class...",
            n_samples
        );
        let result =
            run_incremental_experiment::<Backend>(&dataset, 30, 1, n_samples, epochs, seed)?;
        println!(
            "    New class: {:.2}%, Base after: {:.2}%, Forgetting: {:.2}%",
            result.new_class_accuracy, result.base_accuracy_after, result.forgetting
        );
        large_base_results.push(PositionLabelResult {
            base_classes: 30,
            labeled_samples: n_samples,
            new_class_accuracy: result.new_class_accuracy,
            base_accuracy_after: result.base_accuracy_after,
            forgetting: result.forgetting,
            overall_accuracy: result.overall_accuracy,
            training_time: result.training_time,
        });
    }
    println!();

    // Calculate summary statistics
    let summary = calculate_position_summary(&small_base_results, &large_base_results);

    let results = NewClassPositionResults {
        small_base_results,
        large_base_results,
        summary,
    };

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_new_class_position_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG charts
    generate_new_class_position_charts(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Calculate summary statistics for the position experiment
fn calculate_position_summary(
    small_results: &[PositionLabelResult],
    large_results: &[PositionLabelResult],
) -> PositionSummary {
    // Find minimum samples for 70% accuracy
    let min_samples_small_70pct = small_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 70.0)
        .map(|r| r.labeled_samples)
        .min();

    let min_samples_large_70pct = large_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 70.0)
        .map(|r| r.labeled_samples)
        .min();

    // Find minimum samples for 80% accuracy
    let min_samples_small_80pct = small_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 80.0)
        .map(|r| r.labeled_samples)
        .min();

    let min_samples_large_80pct = large_results
        .iter()
        .filter(|r| r.new_class_accuracy >= 80.0)
        .map(|r| r.labeled_samples)
        .min();

    // Calculate average forgetting difference
    let avg_small_forgetting: f64 = if !small_results.is_empty() {
        small_results.iter().map(|r| r.forgetting).sum::<f64>() / small_results.len() as f64
    } else {
        0.0
    };

    let avg_large_forgetting: f64 = if !large_results.is_empty() {
        large_results.iter().map(|r| r.forgetting).sum::<f64>() / large_results.len() as f64
    } else {
        0.0
    };

    let avg_forgetting_difference = avg_large_forgetting - avg_small_forgetting;

    // Compare equivalent accuracy levels to determine if harder as 31st
    let harder_as_31st = match (min_samples_small_70pct, min_samples_large_70pct) {
        (Some(small), Some(large)) => large > small,
        (Some(_), None) => true, // Large never achieved 70%
        _ => false,
    };

    // Calculate samples ratio for equivalent accuracy
    let samples_ratio = match (min_samples_small_70pct, min_samples_large_70pct) {
        (Some(small), Some(large)) if small > 0 => large as f64 / small as f64,
        _ => 1.0,
    };

    PositionSummary {
        min_samples_small_70pct,
        min_samples_large_70pct,
        min_samples_small_80pct,
        min_samples_large_80pct,
        avg_forgetting_difference,
        harder_as_31st,
        samples_ratio,
    }
}

/// Generate conclusions for new class position experiment
fn generate_new_class_position_conclusions(results: &NewClassPositionResults) -> String {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");

    let mut text = String::new();
    text.push_str("========================================================================\n");
    text.push_str("EXPERIMENT 4: New Class Position Effect - Conclusions\n");
    text.push_str(&format!("Generated: {}\n", timestamp));
    text.push_str("========================================================================\n\n");

    text.push_str("RESEARCH QUESTION:\n");
    text.push_str(
        "Does adding a class as the 6th class (small base) require different amounts of\n",
    );
    text.push_str("labeled samples compared to adding as the 31st class (large base)?\n\n");

    text.push_str("RESULTS:\n\n");

    // Table header
    text.push_str(&format!(
        "{:>8} | {:>12} | {:>12} | {:>12} | {:>12}\n",
        "Labels", "6th Class", "31st Class", "Difference", "Ratio"
    ));
    text.push_str(&format!("{}\n", "-".repeat(68)));

    // Compare results at each label count
    for (small, large) in results
        .small_base_results
        .iter()
        .zip(results.large_base_results.iter())
    {
        let diff = large.new_class_accuracy - small.new_class_accuracy;
        let ratio = if small.new_class_accuracy > 0.0 {
            large.new_class_accuracy / small.new_class_accuracy
        } else {
            0.0
        };
        text.push_str(&format!(
            "{:>8} | {:>11.2}% | {:>11.2}% | {:>+11.2}% | {:>11.2}x\n",
            small.labeled_samples, small.new_class_accuracy, large.new_class_accuracy, diff, ratio
        ));
    }

    text.push_str("\nFORGETTING ANALYSIS:\n\n");
    text.push_str(&format!(
        "{:>8} | {:>12} | {:>12} | {:>12}\n",
        "Labels", "5→6 Forget", "30→31 Forget", "Difference"
    ));
    text.push_str(&format!("{}\n", "-".repeat(56)));

    for (small, large) in results
        .small_base_results
        .iter()
        .zip(results.large_base_results.iter())
    {
        let diff = large.forgetting - small.forgetting;
        text.push_str(&format!(
            "{:>8} | {:>11.2}% | {:>11.2}% | {:>+11.2}%\n",
            small.labeled_samples, small.forgetting, large.forgetting, diff
        ));
    }

    text.push_str("\nKEY FINDINGS:\n");

    let summary = &results.summary;
    match summary.min_samples_small_70pct {
        Some(min) => text.push_str(&format!(
            "1. Minimum samples for 70% accuracy (6th class): {} images\n",
            min
        )),
        None => text.push_str(
            "1. 70% accuracy not reached for the 6th class with the tested sample counts\n",
        ),
    }
    match summary.min_samples_large_70pct {
        Some(min) => text.push_str(&format!(
            "   Minimum samples for 70% accuracy (31st class): {} images\n",
            min
        )),
        None => text.push_str(
            "   70% accuracy not reached for the 31st class with the tested sample counts\n",
        ),
    }
    if let (Some(small), Some(large)) = (
        summary.min_samples_small_70pct,
        summary.min_samples_large_70pct,
    ) {
        text.push_str(&format!(
            "   The 31st class needed {:.2}x the samples of the 6th class ({} vs {})\n",
            summary.samples_ratio, large, small
        ));
    }

    text.push_str(&format!(
        "2. Average forgetting, 30→31 minus 5→6: {:+.2}% points\n",
        summary.avg_forgetting_difference
    ));

    text
}

/// Generate SVG charts for new class position experiment
fn generate_new_class_position_charts(
    results: &NewClassPositionResults,
    output_dir: &str,
) -> Result<()> {
    use plantvillage_ssl::utils::charts::{generate_line_chart, DataPoint, DataSeries};

    // Line chart: New class accuracy vs labeled samples for both positions
    let series = vec![
        DataSeries {
            name: "6th Class (5 base)".to_string(),
            points: results
                .small_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.new_class_accuracy,
                    label: Some(format!("{:.1}%", r.new_class_accuracy)),
                })
                .collect(),
            color: "#3498db".to_string(),
        },
        DataSeries {
            name: "31st Class (30 base)".to_string(),
            points: results
                .large_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.new_class_accuracy,
                    label: Some(format!("{:.1}%", r.new_class_accuracy)),
                })
                .collect(),
            color: "#e74c3c".to_string(),
        },
    ];

    let chart_path = Path::new(output_dir).join("new_class_accuracy_curve.svg");
    generate_line_chart(
        "New Class Accuracy: 6th vs 31st Class Position",
        "Labeled Samples",
        "New Class Accuracy (%)",
        &series,
        &chart_path,
    )?;
    println!("New class accuracy chart saved to: {:?}", chart_path);

    // Line chart: Forgetting vs labeled samples
    let forgetting_series = vec![
        DataSeries {
            name: "5→6 Forgetting".to_string(),
            points: results
                .small_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.forgetting,
                    label: Some(format!("{:.1}%", r.forgetting)),
                })
                .collect(),
            color: "#2ecc71".to_string(),
        },
        DataSeries {
            name: "30→31 Forgetting".to_string(),
            points: results
                .large_base_results
                .iter()
                .map(|r| DataPoint {
                    x: r.labeled_samples as f64,
                    y: r.forgetting,
                    label: Some(format!("{:.1}%", r.forgetting)),
                })
                .collect(),
            color: "#9b59b6".to_string(),
        },
    ];

    let forgetting_path = Path::new(output_dir).join("forgetting_curve.svg");
    generate_line_chart(
        "Forgetting: 5→6 vs 30→31 Classes",
        "Labeled Samples",
        "Forgetting (%)",
        &forgetting_series,
        &forgetting_path,
    )?;
    println!("Forgetting chart saved to: {:?}", forgetting_path);

    // Generate comparison bar chart at 50 samples
    let fifty_sample_small = results
        .small_base_results
        .iter()
        .find(|r| r.labeled_samples == 50);
    let fifty_sample_large = results
        .large_base_results
        .iter()
        .find(|r| r.labeled_samples == 50);

    if let (Some(small), Some(large)) = (fifty_sample_small, fifty_sample_large) {
        let comparison_path = Path::new(output_dir).join("position_comparison_50.svg");
        generate_position_comparison_chart(
            "New Class Position Effect (50 labeled samples)",
            small,
            large,
            &comparison_path,
        )?;
        println!("Position comparison chart saved to: {:?}", comparison_path);
    }

    Ok(())
}

/// Generate a comparison chart for position experiment
fn generate_position_comparison_chart(
    title: &str,
    small: &PositionLabelResult,
    large: &PositionLabelResult,
    output_path: &Path,
) -> std::io::Result<()> {
    const CHART_WIDTH: f64 = 900.0;
    const CHART_HEIGHT: f64 = 500.0;
    const MARGIN_TOP: f64 = 70.0;
    const MARGIN_RIGHT: f64 = 40.0;
    const MARGIN_BOTTOM: f64 = 100.0;
    const MARGIN_LEFT: f64 = 80.0;
    const COLOR_GRID: &str = "#ecf0f1";
    const COLOR_TEXT: &str = "#2c3e50";

    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    let metrics = [
        (
            "New Class Acc",
            small.new_class_accuracy,
            large.new_class_accuracy,
            "#3498db",
            "#e74c3c",
        ),
        (
            "Base After",
            small.base_accuracy_after,
            large.base_accuracy_after,
            "#2ecc71",
            "#27ae60",
        ),
        (
            "Overall",
            small.overall_accuracy,
            large.overall_accuracy,
            "#9b59b6",
            "#8e44ad",
        ),
        (
            "Forgetting",
            small.forgetting,
            large.forgetting,
            "#f39c12",
            "#d35400",
        ),
    ];

    let y_max = 100.0;
    let group_width = plot_width / metrics.len() as f64;
    let bar_width = group_width * 0.35;

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
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:.0}%</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Bars
    for (i, (name, small_val, large_val, small_color, large_color)) in metrics.iter().enumerate() {
        let group_x = MARGIN_LEFT + (i as f64 * group_width) + group_width * 0.1;

        // Small base bar
        let bar_height = (small_val.abs() / y_max) * plot_height;
        let y = MARGIN_TOP + plot_height - bar_height;
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            group_x,
            y,
            bar_width,
            bar_height.max(2.0),
            small_color
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="{}">{:.1}%</text>"#,
            group_x + bar_width / 2.0, y - 5.0, COLOR_TEXT, small_val
        ));

        // Large base bar
        let bar_height = (large_val.abs() / y_max) * plot_height;
        let y = MARGIN_TOP + plot_height - bar_height;
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="4"/>"#,
            group_x + bar_width + 5.0,
            y,
            bar_width,
            bar_height.max(2.0),
            large_color
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="{}">{:.1}%</text>"#,
            group_x + bar_width * 1.5 + 5.0, y - 5.0, COLOR_TEXT, large_val
        ));

        // Label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="{}">{}</text>"#,
            group_x + bar_width + 2.5, CHART_HEIGHT - 60.0, COLOR_TEXT, name
        ));
    }

    // Legend
    let legend_y = CHART_HEIGHT - 30.0;
    svg.push_str(&format!(
        r##"<rect x="{}" y="{}" width="15" height="15" fill="#3498db" rx="2"/>"##,
        CHART_WIDTH / 2.0 - 120.0,
        legend_y - 12.0
    ));
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">6th Class (5 base)</text>"#,
        CHART_WIDTH / 2.0 - 100.0, legend_y, COLOR_TEXT
    ));
    svg.push_str(&format!(
        r##"<rect x="{}" y="{}" width="15" height="15" fill="#e74c3c" rx="2"/>"##,
        CHART_WIDTH / 2.0 + 50.0,
        legend_y - 12.0
    ));
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">31st Class (30 base)</text>"#,
        CHART_WIDTH / 2.0 + 70.0, legend_y, COLOR_TEXT
    ));

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn position(base_classes: usize, labeled_samples: usize, acc: f64) -> PositionLabelResult {
        PositionLabelResult {
            base_classes,
            labeled_samples,
            new_class_accuracy: acc,
            base_accuracy_after: 90.0,
            forgetting: 1.0,
            overall_accuracy: 85.0,
            training_time: 1.0,
        }
    }

    #[test]
    fn test_new_class_position_text_follows_numbers() {
        let small = vec![position(5, 10, 60.0), position(5, 50, 75.0)];
        let large = vec![position(30, 10, 50.0), position(30, 50, 65.0)];
        let summary = calculate_position_summary(&small, &large);
        let results = NewClassPositionResults {
            small_base_results: small,
            large_base_results: large,
            summary,
        };
        let text = generate_new_class_position_conclusions(&results);
        assert!(text.contains("Minimum samples for 70% accuracy (6th class): 50 images"));
        assert!(text.contains("70% accuracy not reached for the 31st class"));
        assert!(!text.contains("x the samples"));
        assert!(!text.contains("SSL"));
    }
}
