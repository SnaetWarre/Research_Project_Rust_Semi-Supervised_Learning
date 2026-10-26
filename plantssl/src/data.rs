//! Dataset scanning and the fixed train/stream/validation/test split.
//!
//! The Kaggle "New Plant Diseases" dataset was built by rotating and flipping each original
//! PlantVillage photo several times and saving every copy as its own file. A random per-image
//! split therefore puts copies of the same leaf in both training and test data. The split here
//! groups files by their source photo and keeps every group inside a single pool.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// One image and its class index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sample {
    /// Path relative to the dataset root.
    pub path: PathBuf,
    pub label: usize,
}

/// All images found under a dataset root, with classes in sorted order.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub root: PathBuf,
    pub classes: Vec<String>,
    pub samples: Vec<Sample>,
}

impl Dataset {
    /// Scans `root/<class>/*` or, for the Kaggle layout, `root/{train,valid}/<class>/*`.
    pub fn scan(root: &Path) -> Result<Self> {
        // Absolute, so the saved split works from any working directory
        let root = &fs::canonicalize(root)
            .with_context(|| format!("dataset folder {} not found", root.display()))?;
        let parts: Vec<PathBuf> = ["train", "valid"]
            .iter()
            .map(|part| root.join(part))
            .filter(|dir| dir.is_dir())
            .collect();
        let parts = if parts.is_empty() {
            vec![root.to_path_buf()]
        } else {
            parts
        };

        let mut by_class: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
        for part in &parts {
            for class_dir in sorted_entries(part)? {
                if !class_dir.is_dir() {
                    continue;
                }
                let class = file_name(&class_dir);
                let files = by_class.entry(class).or_default();
                for file in sorted_entries(&class_dir)? {
                    if is_image(&file) {
                        files.push(file.strip_prefix(root)?.to_path_buf());
                    }
                }
            }
        }
        by_class.retain(|_, files| !files.is_empty());
        if by_class.is_empty() {
            bail!("no images found under {}", root.display());
        }

        let classes: Vec<String> = by_class.keys().cloned().collect();
        let samples = by_class
            .into_values()
            .enumerate()
            .flat_map(|(label, files)| files.into_iter().map(move |path| Sample { path, label }))
            .collect();
        Ok(Self {
            root: root.to_path_buf(),
            classes,
            samples,
        })
    }
}

/// Fractions of the whole dataset for each pool. The stream pool gets the rest.
#[derive(Debug, Clone, Copy)]
pub struct SplitConfig {
    pub labeled: f64,
    pub validation: f64,
    pub test: f64,
    pub seed: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            labeled: 0.2,
            validation: 0.1,
            test: 0.1,
            seed: 42,
        }
    }
}

/// The four pools every command shares. Saved once and loaded by `train`, `ssl` and `eval`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    pub data_dir: PathBuf,
    pub seed: u64,
    pub classes: Vec<String>,
    /// Images the supervised model may train on with their labels.
    pub labeled: Vec<Sample>,
    /// Images that arrive without labels during SSL, in arrival order. Their labels are kept
    /// only to measure pseudo-label precision and for the fully supervised upper bound.
    pub stream: Vec<Sample>,
    /// Used for model selection.
    pub validation: Vec<Sample>,
    /// Used once, for the final numbers.
    pub test: Vec<Sample>,
}

impl Split {
    pub fn new(dataset: &Dataset, config: SplitConfig) -> Result<Self> {
        ensure!(
            config.labeled > 0.0
                && config.validation > 0.0
                && config.test > 0.0
                && config.labeled + config.validation + config.test < 1.0,
            "pool fractions must be positive and leave room for the stream pool"
        );

        let mut split = Self {
            data_dir: dataset.root.clone(),
            seed: config.seed,
            classes: dataset.classes.clone(),
            labeled: Vec::new(),
            stream: Vec::new(),
            validation: Vec::new(),
            test: Vec::new(),
        };

        for label in 0..dataset.classes.len() {
            let mut groups: BTreeMap<String, Vec<Sample>> = BTreeMap::new();
            for sample in dataset.samples.iter().filter(|s| s.label == label) {
                groups
                    .entry(source_id(&sample.path))
                    .or_default()
                    .push(sample.clone());
            }
            let mut groups: Vec<Vec<Sample>> = groups.into_values().collect();
            let mut rng = ChaCha8Rng::seed_from_u64(config.seed.wrapping_add(label as u64));
            groups.shuffle(&mut rng);

            let total: usize = groups.iter().map(Vec::len).sum();
            let test_end = (config.test * total as f64).round() as usize;
            let validation_end = test_end + (config.validation * total as f64).round() as usize;
            let labeled_end = validation_end + (config.labeled * total as f64).round() as usize;

            let mut taken = 0;
            for group in groups {
                let pool = if taken < test_end {
                    &mut split.test
                } else if taken < validation_end {
                    &mut split.validation
                } else if taken < labeled_end {
                    &mut split.labeled
                } else {
                    &mut split.stream
                };
                taken += group.len();
                pool.extend(group);
            }
        }

        // The stream arrives in random class order, like photos from a field would.
        split
            .stream
            .shuffle(&mut ChaCha8Rng::seed_from_u64(config.seed));
        Ok(split)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        crate::write_json(path, self)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("cannot read split file {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Absolute paths and labels for a pool.
    pub fn resolve(&self, pool: &[Sample]) -> Vec<(PathBuf, usize)> {
        pool.iter()
            .map(|s| (self.data_dir.join(&s.path), s.label))
            .collect()
    }
}

/// Identifies the original photo a file was made from, so augmented copies share one id.
///
/// `abc___FREC_Scab 3417_90deg.JPG` and `abc___FREC_Scab 3417.JPG` both map to
/// `<class>/abc___FREC_Scab 3417`.
pub fn source_id(path: &Path) -> String {
    let class = path
        .parent()
        .and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let base = match stem.rsplit_once('_') {
        Some((base, suffix)) if is_augmentation_suffix(suffix) => base,
        _ => stem.as_str(),
    };
    format!("{class}/{base}")
}

fn is_augmentation_suffix(suffix: &str) -> bool {
    suffix == "flipLR"
        || suffix == "flipTB"
        || suffix.starts_with("new")
        || suffix
            .strip_suffix("deg")
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

fn sorted_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = fs::read_dir(dir)
        .with_context(|| format!("cannot read {}", dir.display()))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort();
    Ok(entries)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_ascii_lowercase())
        .is_some_and(|e| e == "jpg" || e == "jpeg" || e == "png")
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn source_id_strips_augmentation_suffixes() {
        let id = |p: &str| source_id(Path::new(p));
        let original = id("train/Apple/abc___FREC_Scab 3417.JPG");
        assert_eq!(original, "Apple/abc___FREC_Scab 3417");
        for copy in [
            "valid/Apple/abc___FREC_Scab 3417_90deg.JPG",
            "train/Apple/abc___FREC_Scab 3417_flipLR.JPG",
            "train/Apple/abc___FREC_Scab 3417_new30degFlipLR.JPG",
            "train/Apple/abc___FREC_Scab 3417_newPixel25.JPG",
        ] {
            assert_eq!(id(copy), original, "{copy}");
        }
        assert_eq!(id("train/Corn/RS_Rust 2312.JPG"), "Corn/RS_Rust 2312");
        assert_eq!(
            id("train/Corn/RS_Rust 2312_flipTB.JPG"),
            "Corn/RS_Rust 2312"
        );
    }

    fn fake_dataset() -> Dataset {
        let mut samples = Vec::new();
        for (label, class) in ["a", "b"].iter().enumerate() {
            for leaf in 0..50 {
                for suffix in ["", "_90deg", "_flipLR"] {
                    samples.push(Sample {
                        path: PathBuf::from(format!("train/{class}/leaf{leaf}{suffix}.jpg")),
                        label,
                    });
                }
            }
        }
        Dataset {
            root: PathBuf::from("data"),
            classes: vec!["a".into(), "b".into()],
            samples,
        }
    }

    #[test]
    fn split_keeps_copies_of_a_leaf_in_one_pool() {
        let split = Split::new(&fake_dataset(), SplitConfig::default()).unwrap();
        let pools = [
            &split.labeled,
            &split.stream,
            &split.validation,
            &split.test,
        ];
        let ids: Vec<HashSet<String>> = pools
            .iter()
            .map(|pool| pool.iter().map(|s| source_id(&s.path)).collect())
            .collect();
        for i in 0..ids.len() {
            for j in i + 1..ids.len() {
                assert!(
                    ids[i].is_disjoint(&ids[j]),
                    "pools {i} and {j} share a leaf"
                );
            }
        }
        let total: usize = pools.iter().map(|p| p.len()).sum();
        assert_eq!(total, 300);
        assert_eq!(split.test.len(), 30);
        assert_eq!(split.labeled.len(), 60);
    }

    #[test]
    fn split_is_deterministic() {
        let a = Split::new(&fake_dataset(), SplitConfig::default()).unwrap();
        let b = Split::new(&fake_dataset(), SplitConfig::default()).unwrap();
        assert_eq!(a.test, b.test);
        assert_eq!(a.stream, b.stream);
    }
}
