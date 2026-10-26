# References and Methodology

**Author:** Warre Snaet  
**Project:** Research Project 2025-2026  
**Topic:** Semi-Supervised Plant Disease Classification on Edge Devices

---

## Methodology and Technology Justification

This project integrates state-of-the-art technologies to achieve high-performance edge computing. The choice of technology is motivated by the following requirements:

* **Rust & Burn Framework**: Chosen for memory safety and zero-cost abstractions, essential for resource-constrained edge devices [1], [3]. Unlike Python-based solutions, Rust provides predictable performance without garbage collection pauses [15].
* **Semi-Supervised Learning (SSL)**: Selected to address the scarcity of labeled agricultural data. The approach utilizes "Learning without Forgetting" (LwF) principles to adapt to new data streams [6].
* **Tauri & Svelte**: Provides a lightweight, secure native interface compared to Electron, minimizing resource usage on the host device [4], [5].

---

## IEEE References

### Frameworks and Technical Documentation

[1] Tracel-AI, "Burn: A Flexible and Modular Tensor Library for Machine Learning in Rust," *Burn.dev Official Documentation*, 2025. [Online]. Available: <https://burn.dev/docs/burn/>. [Accessed: Jan. 25, 2026].

[2] Tracel-AI Contributors, "Burn - GitHub Repository," *GitHub*, 2024. [Online]. Available: <https://github.com/tracel-ai/burn>. [Accessed: Jan. 25, 2026].

[3] RantAI, "Deep Learning via Rust: Comparative Analysis of tch-rs and burn," in *DLVR Book*, Chapter 4, 2024. [Online]. Available: <https://dlvr.rantai.dev/docs/part-i/chapter-4/>.

[4] Tauri Contributors, "Tauri - Build Smaller, Faster, and More Secure Desktop Applications," *Tauri Studio*, 2024. [Online]. Available: <https://tauri.app/>.

[5] Svelte Contributors, "Svelte 5 Documentation," *Svelte.dev*, 2024. [Online]. Available: <https://svelte.dev/docs>.

### Academic Studies and Books

[6] Z. Li and D. Hoiem, "Learning without Forgetting," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 40, no. 12, pp. 2935–2947, Dec. 2018.

[7] S. Klabnik and C. Nichols, *The Rust Programming Language*. San Francisco, CA: No Starch Press, 2018. [Online]. Available: <https://doc.rust-lang.org/book/>.

[8] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*. Cambridge, MA: MIT Press, 2016.

### Datasets and Standards

[9] V. Singh, "New Plant Diseases Dataset (PlantVillage Balanced)," *Kaggle Datasets*, 2020. [Online]. Available: <https://www.kaggle.com/datasets/chandraguptsingh/plantvillage-balanced>.

### Hardware and Edge Computing

[10] NVIDIA Corporation, "Jetson Orin Nano Developer Kit," *NVIDIA Embedded Computing*, 2023. [Online]. Available: <https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit>.

[11] NVIDIA Corporation, "CUDA Toolkit Documentation v12.0," *NVIDIA Developer*, 2024. [Online]. Available: <https://docs.nvidia.com/cuda/>.

### Technical Articles and Reports

[12] P. Yaw, "Burn: The Future of Deep Learning in Rust," *Dev.to*, Dec. 2024. [Online]. Available: <https://dev.to/philip_yaw/burn-the-future-of-deep-learning-in-rust-5c5e>.

[13] Calmops, "Burn: A Modern Deep Learning Framework for Rust," *Calmops Engineering Blog*, Dec. 2025. [Online]. Available: <https://calmops.com/programming/rust/burn-framework-rust-ml/>.

[14] Hamze, "Rust Ecosystem for AI & LLMs," *HackMD Technical Notes*, Apr. 2025. [Online]. Available: <https://hackmd.io/@Hamze/Hy5LiRV1gg>.

[15] MarkAICode, "Rust for Machine Learning in 2025: Framework Comparison and Performance Metrics," *MarkAICode*, 2025. [Online]. Available: <https://markaicode.com/rust-machine-learning-framework-comparison-2025/>.

[16] Oven Contributors, "Bun - Fast JavaScript Runtime," *Bun.sh*, 2024. [Online]. Available: <https://bun.sh/>.
