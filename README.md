# Plant Disease Detection - Research Project

**Student:** Warre Snaet | **Institution:** Howest MCT

Semi-supervised learning for plant disease classification on embedded edge devices using Rust.

---

##  Structure

```
Source/
├── plantvillage_ssl/      # SSL implementation (pseudo-labeling)  WORKS
├── incremental_learning/  # Add new classes (5→6, 30→31 experiments)
├── pytorch_reference/     # Python reference for comparison
├── benchmarks/            # Framework comparison scripts
└── research/              # Literature study, contract, meeting notes
```

---

##  Quick Start

### 1. Download Dataset (Once)
```bash
./download_plantvillage.sh
```

### 2. SSL Training
```bash
cd plantvillage_ssl
cargo build --release
./target/release/plantvillage_ssl train \
    --data-dir data/plantvillage \
    --labeled-ratio 0.2 \
    --epochs 30 --cuda
```

### 3. Incremental Learning
```bash
cd incremental_learning
cargo build --release
./target/release/plant-incremental experiment \
    --method lwf \
    --base-classes 5 \
    --new-classes 1 \
    --data-dir ../plantvillage_ssl/data/plantvillage
```

---

##  Research Questions

1. **SSL:** How efficient is pseudo-labeling on edge devices?
2. **Incremental:** Is 5→6 harder than 30→31 classes?
3. **Data efficiency:** How many images needed per new class?

---

##  Documentation

- [plantvillage_ssl/docs/](plantvillage_ssl/docs/) - Installation & user guide
- [research/literatuurstudie.md](research/literatuurstudie.md) - Literature review
