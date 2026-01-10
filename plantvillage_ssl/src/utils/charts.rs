//! SVG Chart Generator for Experiment Results
//!
//! Generates clean SVG charts for visualizing experiment results.
//! These can be used directly in presentations or reports.

use std::fs;
use std::path::Path;

/// Chart styling constants
const CHART_WIDTH: f64 = 800.0;
const CHART_HEIGHT: f64 = 500.0;
const MARGIN_TOP: f64 = 60.0;
const MARGIN_RIGHT: f64 = 40.0;
const MARGIN_BOTTOM: f64 = 80.0;
const MARGIN_LEFT: f64 = 80.0;

const COLOR_PRIMARY: &str = "#3498db";
const COLOR_SECONDARY: &str = "#2ecc71";
const COLOR_TERTIARY: &str = "#e74c3c";
const COLOR_GRID: &str = "#ecf0f1";
const COLOR_AXIS: &str = "#2c3e50";
const COLOR_TEXT: &str = "#2c3e50";

/// A data point for a line chart
#[derive(Debug, Clone)]
pub struct DataPoint {
    pub x: f64,
    pub y: f64,
    pub label: Option<String>,
}

/// A data series for charts
#[derive(Debug, Clone)]
pub struct DataSeries {
    pub name: String,
    pub points: Vec<DataPoint>,
    pub color: String,
}

/// Bar chart data
#[derive(Debug, Clone)]
pub struct BarData {
    pub label: String,
    pub value: f64,
    pub color: String,
}

/// Generate a line chart SVG
pub fn generate_line_chart(
    title: &str,
    x_label: &str,
    y_label: &str,
    series: &[DataSeries],
    output_path: &Path,
) -> std::io::Result<()> {
    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    // Find data ranges
    let (x_min, x_max, y_min, y_max) = find_ranges(series);
    let y_min = 0.0; // Always start Y at 0 for accuracy charts
    let y_max = 100.0_f64.max(y_max); // Cap at 100 for percentage

    let mut svg = String::new();

    // SVG header
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        CHART_WIDTH, CHART_HEIGHT, CHART_WIDTH, CHART_HEIGHT
    ));

    // Background
    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        CHART_WIDTH, CHART_HEIGHT
    ));

    // Title
    svg.push_str(&format!(
        r#"<text x="{}" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
        CHART_WIDTH / 2.0, COLOR_TEXT, escape_xml(title)
    ));

    // Grid lines
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = y_min + (i as f64 / 5.0) * (y_max - y_min);
        
        // Grid line
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT, y, MARGIN_LEFT + plot_width, y, COLOR_GRID
        ));
        
        // Y-axis label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:.0}%</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Axes
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT, MARGIN_TOP + plot_height, MARGIN_LEFT + plot_width, MARGIN_TOP + plot_height, COLOR_AXIS
    ));
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT, MARGIN_TOP, MARGIN_LEFT, MARGIN_TOP + plot_height, COLOR_AXIS
    ));

    // Axis labels
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{}">{}</text>"#,
        MARGIN_LEFT + plot_width / 2.0, CHART_HEIGHT - 20.0, COLOR_TEXT, escape_xml(x_label)
    ));
    svg.push_str(&format!(
        r#"<text x="20" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{}" transform="rotate(-90 20 {})">{}</text>"#,
        CHART_HEIGHT / 2.0, COLOR_TEXT, CHART_HEIGHT / 2.0, escape_xml(y_label)
    ));

    // Plot each series
    for series_data in series {
        if series_data.points.is_empty() {
            continue;
        }

        // Line path
        let mut path = String::new();
        for (i, point) in series_data.points.iter().enumerate() {
            let x = MARGIN_LEFT + ((point.x - x_min) / (x_max - x_min)) * plot_width;
            let y = MARGIN_TOP + plot_height - ((point.y - y_min) / (y_max - y_min)) * plot_height;
            
            if i == 0 {
                path.push_str(&format!("M {} {}", x, y));
            } else {
                path.push_str(&format!(" L {} {}", x, y));
            }
        }

        svg.push_str(&format!(
            r#"<path d="{}" fill="none" stroke="{}" stroke-width="3"/>"#,
            path, series_data.color
        ));

        // Data points
        for point in &series_data.points {
            let x = MARGIN_LEFT + ((point.x - x_min) / (x_max - x_min)) * plot_width;
            let y = MARGIN_TOP + plot_height - ((point.y - y_min) / (y_max - y_min)) * plot_height;

            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="5" fill="{}" stroke="white" stroke-width="2"/>"#,
                x, y, series_data.color
            ));

            // Point label
            if let Some(label) = &point.label {
                svg.push_str(&format!(
                    r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="{}">{}</text>"#,
                    x, y - 12.0, COLOR_TEXT, escape_xml(label)
                ));
            }
        }
    }

    // X-axis tick labels
    for series_data in series {
        for point in &series_data.points {
            let x = MARGIN_LEFT + ((point.x - x_min) / (x_max - x_min)) * plot_width;
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{}">{:.0}</text>"#,
                x, MARGIN_TOP + plot_height + 20.0, COLOR_TEXT, point.x
            ));
        }
        break; // Only need labels from first series
    }

    // Legend
    let mut legend_y = MARGIN_TOP + 10.0;
    for series_data in series {
        svg.push_str(&format!(
            r#"<rect x="{}" y="{}" width="15" height="15" fill="{}"/>"#,
            CHART_WIDTH - MARGIN_RIGHT - 100.0, legend_y, series_data.color
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="12" fill="{}">{}</text>"#,
            CHART_WIDTH - MARGIN_RIGHT - 80.0, legend_y + 12.0, COLOR_TEXT, escape_xml(&series_data.name)
        ));
        legend_y += 25.0;
    }

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}

/// Generate a bar chart SVG
pub fn generate_bar_chart(
    title: &str,
    y_label: &str,
    bars: &[BarData],
    output_path: &Path,
) -> std::io::Result<()> {
    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    let y_max = bars.iter().map(|b| b.value).fold(0.0f64, f64::max).max(100.0);

    let bar_width = (plot_width / bars.len() as f64) * 0.7;
    let bar_gap = (plot_width / bars.len() as f64) * 0.3;

    let mut svg = String::new();

    // SVG header
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}" width="{}" height="{}">"#,
        CHART_WIDTH, CHART_HEIGHT, CHART_WIDTH, CHART_HEIGHT
    ));

    // Background
    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        CHART_WIDTH, CHART_HEIGHT
    ));

    // Title
    svg.push_str(&format!(
        r#"<text x="{}" y="35" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="{}">{}</text>"#,
        CHART_WIDTH / 2.0, COLOR_TEXT, escape_xml(title)
    ));

    // Grid lines
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = (i as f64 / 5.0) * y_max;
        
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT, y, MARGIN_LEFT + plot_width, y, COLOR_GRID
        ));
        
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:.0}%</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Axes
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT, MARGIN_TOP + plot_height, MARGIN_LEFT + plot_width, MARGIN_TOP + plot_height, COLOR_AXIS
    ));

    // Y-axis label
    svg.push_str(&format!(
        r#"<text x="20" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="{}" transform="rotate(-90 20 {})">{}</text>"#,
        CHART_HEIGHT / 2.0, COLOR_TEXT, CHART_HEIGHT / 2.0, escape_xml(y_label)
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

        // Value label on top
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="{}">{:.1}%</text>"#,
            x + bar_width / 2.0, y - 8.0, COLOR_TEXT, bar.value
        ));

        // X-axis label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="{}">{}</text>"#,
            x + bar_width / 2.0, MARGIN_TOP + plot_height + 25.0, COLOR_TEXT, escape_xml(&bar.label)
        ));
    }

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}

/// Generate a grouped bar chart for comparison (e.g., 5→6 vs 30→31)
pub fn generate_comparison_chart(
    title: &str,
    groups: &[(&str, Vec<(&str, f64, &str)>)], // (group_label, [(bar_label, value, color)])
    output_path: &Path,
) -> std::io::Result<()> {
    let plot_width = CHART_WIDTH - MARGIN_LEFT - MARGIN_RIGHT;
    let plot_height = CHART_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM;

    let y_max = groups
        .iter()
        .flat_map(|(_, bars)| bars.iter().map(|(_, v, _)| *v))
        .fold(0.0f64, f64::max)
        .max(100.0);

    let group_width = plot_width / groups.len() as f64;
    let bars_per_group = groups.get(0).map(|(_, b)| b.len()).unwrap_or(0);
    let bar_width = (group_width * 0.8) / bars_per_group as f64;
    let group_padding = group_width * 0.1;

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
        CHART_WIDTH / 2.0, COLOR_TEXT, escape_xml(title)
    ));

    // Grid
    for i in 0..=5 {
        let y = MARGIN_TOP + plot_height - (i as f64 / 5.0) * plot_height;
        let value = (i as f64 / 5.0) * y_max;
        
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="1"/>"#,
            MARGIN_LEFT, y, MARGIN_LEFT + plot_width, y, COLOR_GRID
        ));
        
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="{}">{:.0}%</text>"#,
            MARGIN_LEFT - 10.0, y + 4.0, COLOR_TEXT, value
        ));
    }

    // Axes
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
        MARGIN_LEFT, MARGIN_TOP + plot_height, MARGIN_LEFT + plot_width, MARGIN_TOP + plot_height, COLOR_AXIS
    ));

    // Groups
    for (group_idx, (group_label, bars)) in groups.iter().enumerate() {
        let group_x = MARGIN_LEFT + group_idx as f64 * group_width + group_padding;

        for (bar_idx, (label, value, color)) in bars.iter().enumerate() {
            let x = group_x + bar_idx as f64 * bar_width;
            let bar_height = (*value / y_max) * plot_height;
            let y = MARGIN_TOP + plot_height - bar_height;

            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="{}" height="{}" fill="{}" rx="3"/>"#,
                x, y, bar_width * 0.9, bar_height, color
            ));

            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" font-weight="bold" fill="{}">{:.1}%</text>"#,
                x + bar_width * 0.45, y - 5.0, COLOR_TEXT, value
            ));
        }

        // Group label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="{}">{}</text>"#,
            group_x + (bars.len() as f64 * bar_width) / 2.0, MARGIN_TOP + plot_height + 25.0, COLOR_TEXT, escape_xml(group_label)
        ));
    }

    // Legend
    if let Some((_, first_bars)) = groups.first() {
        let mut legend_x = MARGIN_LEFT;
        for (label, _, color) in first_bars {
            svg.push_str(&format!(
                r#"<rect x="{}" y="{}" width="12" height="12" fill="{}"/>"#,
                legend_x, CHART_HEIGHT - 35.0, color
            ));
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" font-family="Arial, sans-serif" font-size="11" fill="{}">{}</text>"#,
                legend_x + 18.0, CHART_HEIGHT - 25.0, COLOR_TEXT, escape_xml(label)
            ));
            legend_x += 120.0;
        }
    }

    svg.push_str("</svg>");

    fs::write(output_path, svg)
}

fn find_ranges(series: &[DataSeries]) -> (f64, f64, f64, f64) {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for s in series {
        for p in &s.points {
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }
    }

    (x_min, x_max, y_min, y_max)
}

fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_line_chart_generation() {
        let series = vec![DataSeries {
            name: "Test".to_string(),
            points: vec![
                DataPoint { x: 5.0, y: 36.84, label: Some("36.8%".to_string()) },
                DataPoint { x: 10.0, y: 61.84, label: Some("61.8%".to_string()) },
                DataPoint { x: 25.0, y: 65.79, label: Some("65.8%".to_string()) },
            ],
            color: COLOR_PRIMARY.to_string(),
        }];

        let path = PathBuf::from("/tmp/test_chart.svg");
        generate_line_chart("Test Chart", "X Axis", "Y Axis", &series, &path).unwrap();
        assert!(path.exists());
    }
}
