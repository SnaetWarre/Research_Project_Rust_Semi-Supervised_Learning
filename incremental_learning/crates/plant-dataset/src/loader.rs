//! Image loading functionality for plant dataset.
//!
//! This module provides efficient image loading from disk with error handling.

use image::DynamicImage;
use plant_core::{Error, ImageSample, Result};
use std::path::{Path, PathBuf};

/// Image loader for plant dataset
pub struct ImageLoader {
    /// Root directory containing images
    root_dir: PathBuf,
}

impl ImageLoader {
    /// Creates a new image loader
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
        }
    }

    /// Loads an image from a sample
    pub fn load_sample(&self, sample: &ImageSample) -> Result<DynamicImage> {
        self.load_image(&sample.path)
    }

    /// Loads an image from a path
    pub fn load_image(&self, path: &Path) -> Result<DynamicImage> {
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root_dir.join(path)
        };

        if !full_path.exists() {
            return Err(Error::NotFound(format!(
                "Image file not found: {}",
                full_path.display()
            )));
        }

        image::open(&full_path)
            .map_err(|e| Error::Image(format!("Failed to load image {}: {}", full_path.display(), e)))
    }

    /// Checks if an image file exists
    pub fn exists(&self, path: &Path) -> bool {
        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root_dir.join(path)
        };

        full_path.exists()
    }

    /// Gets the full path for an image
    pub fn full_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root_dir.join(path)
        }
    }

    /// Scans a directory for image files
    pub fn scan_directory(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let full_dir = if dir.is_absolute() {
            dir.to_path_buf()
        } else {
            self.root_dir.join(dir)
        };

        if !full_dir.exists() {
            return Err(Error::NotFound(format!(
                "Directory not found: {}",
                full_dir.display()
            )));
        }

        if !full_dir.is_dir() {
            return Err(Error::InvalidArgument(format!(
                "Path is not a directory: {}",
                full_dir.display()
            )));
        }

        let mut images = Vec::new();

        for entry in std::fs::read_dir(&full_dir)
            .map_err(|e| Error::Io(e))?
        {
            let entry = entry.map_err(|e| Error::Io(e))?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if matches!(ext_lower.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif") {
                        images.push(path);
                    }
                }
            }
        }

        Ok(images)
    }

    /// Recursively scans directories for image files
    pub fn scan_directory_recursive(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let full_dir = if dir.is_absolute() {
            dir.to_path_buf()
        } else {
            self.root_dir.join(dir)
        };

        if !full_dir.exists() {
            return Err(Error::NotFound(format!(
                "Directory not found: {}",
                full_dir.display()
            )));
        }

        let mut images = Vec::new();
        self.scan_recursive_helper(&full_dir, &mut images)?;

        Ok(images)
    }

    /// Helper function for recursive directory scanning
    fn scan_recursive_helper(&self, dir: &Path, images: &mut Vec<PathBuf>) -> Result<()> {
        for entry in std::fs::read_dir(dir)
            .map_err(|e| Error::Io(e))?
        {
            let entry = entry.map_err(|e| Error::Io(e))?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    let ext_lower = ext.to_string_lossy().to_lowercase();
                    if matches!(ext_lower.as_str(), "jpg" | "jpeg" | "png" | "bmp" | "gif") {
                        images.push(path);
                    }
                }
            } else if path.is_dir() {
                self.scan_recursive_helper(&path, images)?;
            }
        }

        Ok(())
    }

    /// Loads images from a directory with class label
    pub fn load_class_directory(
        &self,
        class_dir: &Path,
        class_id: usize,
        class_name: Option<String>,
    ) -> Result<Vec<ImageSample>> {
        let images = self.scan_directory(class_dir)?;

        let samples = images
            .into_iter()
            .map(|path| {
                if let Some(ref name) = class_name {
                    ImageSample::with_class_name(path, class_id, name.clone())
                } else {
                    ImageSample::new(path, class_id)
                }
            })
            .collect();

        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn create_test_image(path: &Path) {
        // Create a simple 10x10 red image
        let img = image::ImageBuffer::from_fn(10, 10, |_, _| image::Rgb([255u8, 0u8, 0u8]));
        img.save(path).unwrap();
    }

    #[test]
    fn test_loader_creation() {
        let loader = ImageLoader::new("/tmp");
        assert_eq!(loader.root_dir, PathBuf::from("/tmp"));
    }

    #[test]
    fn test_full_path_relative() {
        let loader = ImageLoader::new("/data");
        let full = loader.full_path(Path::new("images/test.jpg"));
        assert_eq!(full, PathBuf::from("/data/images/test.jpg"));
    }

    #[test]
    fn test_full_path_absolute() {
        let loader = ImageLoader::new("/data");
        let full = loader.full_path(Path::new("/absolute/path.jpg"));
        assert_eq!(full, PathBuf::from("/absolute/path.jpg"));
    }

    #[test]
    fn test_load_image_not_found() {
        let loader = ImageLoader::new("/tmp");
        let result = loader.load_image(Path::new("nonexistent.jpg"));
        assert!(result.is_err());
    }

    #[test]
    fn test_scan_directory() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ImageLoader::new(temp_dir.path());

        // Create test images
        create_test_image(&temp_dir.path().join("image1.jpg"));
        create_test_image(&temp_dir.path().join("image2.png"));
        fs::write(temp_dir.path().join("not_image.txt"), "text").unwrap();

        let images = loader.scan_directory(Path::new("")).unwrap();
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_scan_directory_recursive() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ImageLoader::new(temp_dir.path());

        // Create nested structure
        let sub_dir = temp_dir.path().join("subdir");
        fs::create_dir(&sub_dir).unwrap();

        create_test_image(&temp_dir.path().join("image1.jpg"));
        create_test_image(&sub_dir.join("image2.jpg"));

        let images = loader.scan_directory_recursive(Path::new("")).unwrap();
        assert_eq!(images.len(), 2);
    }

    #[test]
    fn test_load_class_directory() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ImageLoader::new(temp_dir.path());

        create_test_image(&temp_dir.path().join("img1.jpg"));
        create_test_image(&temp_dir.path().join("img2.jpg"));

        let samples = loader
            .load_class_directory(Path::new(""), 0, Some("TestClass".to_string()))
            .unwrap();

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].label, 0);
        assert_eq!(samples[0].class_name, Some("TestClass".to_string()));
    }

    #[test]
    fn test_exists() {
        let temp_dir = TempDir::new().unwrap();
        let loader = ImageLoader::new(temp_dir.path());

        let image_path = temp_dir.path().join("test.jpg");
        create_test_image(&image_path);

        assert!(loader.exists(Path::new("test.jpg")));
        assert!(!loader.exists(Path::new("nonexistent.jpg")));
    }
}
