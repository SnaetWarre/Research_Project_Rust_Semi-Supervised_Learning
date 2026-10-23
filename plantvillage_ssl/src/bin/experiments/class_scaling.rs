//! Experiment 2: class scaling effect.
//!
//! Adds one class to a 5-class and a 30-class base model and compares
//! forgetting and new-class accuracy.

use std::fs;
use std::path::Path;

use anyhow::Result;
use chrono::Local;
use colored::Colorize;
use serde::{Deserialize, Serialize};

use plantvillage_ssl::PlantVillageDataset;

use super::incremental::{run_incremental_experiment, ScalingResult};
use super::Backend;

/// Results from class scaling experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ClassScalingResults {
    /// Results for small base (5 classes + 1)
    small_base: ScalingResult,
    /// Results for large base (30 classes + 1)
    large_base: ScalingResult,
    /// Relative difficulty (large_base forgetting / small_base forgetting)
    relative_difficulty: f64,
}

/// Run class scaling experiment
pub(crate) fn run_class_scaling(
    data_dir: &str,
    output_dir: &str,
    epochs: usize,
    seed: u64,
) -> Result<()> {
    println!("{}", "EXPERIMENT 2: Class Scaling Effect".yellow().bold());
    println!("Question: Is adding a class to 5 classes harder than adding to 30 classes?");
    println!();

    fs::create_dir_all(output_dir)?;

    // Load dataset
    println!("{}", "Loading dataset...".cyan());
    let dataset = PlantVillageDataset::new(data_dir)?;
    let stats = dataset.get_stats();
    println!("  Total samples: {}", stats.total_samples);
    println!("  Classes: {}", stats.num_classes);
    println!();

    // Experiment A: 5 → 6 classes
    println!("{}", "Part A: 5 → 6 classes".yellow().bold());
    let small_base =
        run_incremental_experiment::<Backend>(&dataset, 5, 1, usize::MAX, epochs, seed)?;
    println!(
        "  Base accuracy: {:.2}% → {:.2}% (forgetting: {:.2}%)",
        small_base.base_accuracy_before, small_base.base_accuracy_after, small_base.forgetting
    );
    println!(
        "  New class accuracy: {:.2}%",
        small_base.new_class_accuracy
    );
    println!("  Overall accuracy: {:.2}%", small_base.overall_accuracy);
    println!();

    // Experiment B: 30 → 31 classes
    println!("{}", "Part B: 30 → 31 classes".yellow().bold());
    let large_base =
        run_incremental_experiment::<Backend>(&dataset, 30, 1, usize::MAX, epochs, seed)?;
    println!(
        "  Base accuracy: {:.2}% → {:.2}% (forgetting: {:.2}%)",
        large_base.base_accuracy_before, large_base.base_accuracy_after, large_base.forgetting
    );
    println!(
        "  New class accuracy: {:.2}%",
        large_base.new_class_accuracy
    );
    println!("  Overall accuracy: {:.2}%", large_base.overall_accuracy);
    println!();

    let relative_difficulty = if small_base.forgetting > 0.0 {
        large_base.forgetting / small_base.forgetting
    } else {
        1.0
    };

    let results = ClassScalingResults {
        small_base,
        large_base,
        relative_difficulty,
    };

    // Save results
    let results_path = Path::new(output_dir).join("results.json");
    let json = serde_json::to_string_pretty(&results)?;
    fs::write(&results_path, &json)?;
    println!("Results saved to: {:?}", results_path);

    // Generate conclusions
    let conclusions = generate_class_scaling_conclusions(&results);
    let conclusions_path = Path::new(output_dir).join("conclusions.txt");
    fs::write(&conclusions_path, &conclusions)?;
    println!("Conclusions saved to: {:?}", conclusions_path);

    // Generate SVG chart
    generate_class_scaling_chart(&results, output_dir)?;

    // Print summary
    println!();
    println!("{}", "SUMMARY".green().bold());
    println!("{}", "=".repeat(60));
    print!("{}", conclusions);

    Ok(())
}

/// Generate conclusions for class scaling experiment
fn generate_class_scaling_conclusions(results: &ClassScalingResults) -> String {
    let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S");

    let mut text = String::new();
    text.push_str("========================================================================\n");
    text.push_str("EXPERIMENT 2: Class Scaling Effect - Conclusions\n");
    text.push_str(&format!("Generated: {}\n", timestamp));
    text.push_str("========================================================================\n\n");

    text.push_str("RESEARCH QUESTION:\n");
    text.push_str("Is adding a class to 5 classes harder than adding to 30 classes?\n");
    text.push_str(
        "Does the model become more biased toward existing classes with a larger base?\n\n",
    );

    text.push_str("RESULTS:\n\n");

    text.push_str(&format!(
        "Scenario A: {} → {} classes\n",
        results.small_base.base_classes, results.small_base.total_classes
    ));
    text.push_str(&format!(
        "  Base accuracy:      {:.2}% → {:.2}%\n",
        results.small_base.base_accuracy_before, results.small_base.base_accuracy_after
    ));
    text.push_str(&format!(
        "  New class accuracy: {:.2}%\n",
        results.small_base.new_class_accuracy
    ));
    text.push_str(&format!(
        "  Overall accuracy:   {:.2}%\n",
        results.small_base.overall_accuracy
    ));
    text.push_str(&format!(
        "  Forgetting:         {:.2}% points\n",
        results.small_base.forgetting
    ));
    text.push('\n');

    text.push_str(&format!(
        "Scenario B: {} → {} classes\n",
        results.large_base.base_classes, results.large_base.total_classes
    ));
    text.push_str(&format!(
        "  Base accuracy:      {:.2}% → {:.2}%\n",
        results.large_base.base_accuracy_before, results.large_base.base_accuracy_after
    ));
    text.push_str(&format!(
        "  New class accuracy: {:.2}%\n",
        results.large_base.new_class_accuracy
    ));
    text.push_str(&format!(
        "  Overall accuracy:   {:.2}%\n",
        results.large_base.overall_accuracy
    ));
    text.push_str(&format!(
        "  Forgetting:         {:.2}% points\n",
        results.large_base.forgetting
    ));
    text.push('\n');

    text.push_str("COMPARATIVE ANALYSIS:\n");
    text.push_str(&format!(
        "  Relative difficulty (large/small forgetting): {:.2}x\n",
        results.relative_difficulty
    ));

    let new_class_diff =
        results.large_base.new_class_accuracy - results.small_base.new_class_accuracy;
    text.push_str(&format!(
        "  New class accuracy difference: {:.2}% points\n",
        new_class_diff
    ));

    text.push_str("\nKEY FINDINGS:\n");

    let small = &results.small_base;
    let large = &results.large_base;

    let forgetting_cmp = if large.forgetting > small.forgetting {
        "MORE"
    } else if large.forgetting < small.forgetting {
        "LESS"
    } else {
        "the SAME"
    };
    text.push_str(&format!(
        "1. Adding a class to {} classes caused {} forgetting than adding one to {} ({:.2}% vs {:.2}% points)\n",
        large.base_classes, forgetting_cmp, small.base_classes, large.forgetting, small.forgetting
    ));

    let new_class_cmp = if large.new_class_accuracy > small.new_class_accuracy {
        "HIGHER"
    } else if large.new_class_accuracy < small.new_class_accuracy {
        "LOWER"
    } else {
        "the SAME"
    };
    text.push_str(&format!(
        "2. New class accuracy was {} with the {}-class base ({:.2}% vs {:.2}%)\n",
        new_class_cmp, large.base_classes, large.new_class_accuracy, small.new_class_accuracy
    ));

    text
}

/// Generate SVG chart for class scaling results
fn generate_class_scaling_chart(results: &ClassScalingResults, output_dir: &str) -> Result<()> {
    use plantvillage_ssl::utils::charts::generate_comparison_chart;

    let groups = vec![
        (
            "5 → 6 Classes",
            vec![
                (
                    "Base Before",
                    results.small_base.base_accuracy_before,
                    "#3498db",
                ),
                (
                    "Base After",
                    results.small_base.base_accuracy_after,
                    "#2ecc71",
                ),
                (
                    "New Class",
                    results.small_base.new_class_accuracy,
                    "#9b59b6",
                ),
            ],
        ),
        (
            "30 → 31 Classes",
            vec![
                (
                    "Base Before",
                    results.large_base.base_accuracy_before,
                    "#3498db",
                ),
                (
                    "Base After",
                    results.large_base.base_accuracy_after,
                    "#2ecc71",
                ),
                (
                    "New Class",
                    results.large_base.new_class_accuracy,
                    "#9b59b6",
                ),
            ],
        ),
    ];

    let chart_path = Path::new(output_dir).join("class_scaling_comparison.svg");
    generate_comparison_chart("Class Scaling: 5→6 vs 30→31 Classes", &groups, &chart_path)?;
    println!("Chart saved to: {:?}", chart_path);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scaling(base_classes: usize, new_class_accuracy: f64, forgetting: f64) -> ScalingResult {
        ScalingResult {
            base_classes,
            total_classes: base_classes + 1,
            base_accuracy_before: 90.0,
            base_accuracy_after: 90.0 - forgetting,
            new_class_accuracy,
            overall_accuracy: 85.0,
            forgetting,
            training_time: 1.0,
        }
    }

    #[test]
    fn test_class_scaling_text_follows_numbers() {
        let more = ClassScalingResults {
            small_base: scaling(5, 90.0, 1.0),
            large_base: scaling(30, 80.0, 3.0),
            relative_difficulty: 3.0,
        };
        let text = generate_class_scaling_conclusions(&more);
        assert!(text.contains("caused MORE forgetting"));
        assert!(text.contains("New class accuracy was LOWER"));

        let less = ClassScalingResults {
            small_base: scaling(5, 80.0, 3.0),
            large_base: scaling(30, 90.0, 1.0),
            relative_difficulty: 1.0 / 3.0,
        };
        let text = generate_class_scaling_conclusions(&less);
        assert!(text.contains("caused LESS forgetting"));
        assert!(text.contains("New class accuracy was HIGHER"));
        assert!(!text.contains("RECOMMENDATION"));
    }
}
