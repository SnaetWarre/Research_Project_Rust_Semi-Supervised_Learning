# The Why: Plant Disease Detection on Edge Devices

## The Problem

Imagine a farmer in a remote area. Their crops are showing strange spots on the leaves. Is it a disease? Which one? Should they spray pesticides? Wait it out? By the time they get an expert opinion, it might be too late - the disease has spread, and they've lost half their harvest.

Now imagine they have a simple device - a camera connected to a small computer - that can instantly tell them: "This is Early Blight. Treat it now."

This technology exists. But there's a catch.

## The Current Reality

Today's plant disease detection relies on:
- **Cloud-based AI** - needs internet (many farms don't have it)
- **Expensive hardware** - costs thousands of euros
- **Static models** - can't learn new diseases without a complete rebuild
- **Python/PyTorch** - heavy, slow, needs complex setup on small devices

Farmers in developing countries, remote areas, or even local Belgian greenhouses often can't use these solutions.

## What This Research Asks

**Can we build a plant disease detector that:**
1. Runs on a small, cheap device (no internet needed)?
2. Learns from very little labeled data (labeling is expensive)?
3. Can learn new diseases over time without forgetting old ones?
4. Uses Rust instead of Python for better performance and reliability?

No one has really tried this combination before.

## Why Howest Should Care

### 1. It's Genuinely Novel
Using Rust and the Burn framework for agricultural ML on edge devices is largely unexplored territory. Most researchers default to Python/PyTorch. This project tests whether the Rust ecosystem is ready for real-world ML applications.

### 2. It Solves Real Problems
- **Data scarcity**: Farmers don't have thousands of labeled disease images. Semi-supervised learning makes the most of limited data.
- **Offline operation**: Agricultural settings often lack reliable internet.
- **Adaptability**: New diseases emerge. Climate change brings new pests. A model that can't learn is a model that becomes obsolete.

### 3. It Produces Measurable Results
This isn't theoretical. The research will answer concrete questions:
- How much accuracy do we lose using 30% labeled data vs 100%?
- Which incremental learning method works best: LwF, EWC, Rehearsal, or fine-tuning?
- Is adding class #6 harder than adding class #31?
- Can inference run under 200ms on a Jetson Orin Nano?

### 4. It Has Educational Value
The project combines:
- Deep learning fundamentals
- Systems programming (Rust)
- Edge computing constraints
- Agricultural domain knowledge
- Experimental research methodology

This is the kind of cross-disciplinary work that prepares students for real industry challenges.

## What New Knowledge Will We Gain?

| Question | Why It Matters |
|----------|----------------|
| Can Burn (Rust ML framework) match PyTorch accuracy? | Determines if Rust is viable for ML deployment |
| How effective is pseudo-labeling on edge devices? | Shows if SSL works with hardware constraints |
| Which incremental learning method minimizes forgetting? | Guides future continuous learning systems |
| Is learning 5->6 classes different from 30->31? | Reveals how model capacity affects learning |
| What's the minimum data needed per new class? | Practical guidance for real deployments |

These answers don't exist yet. This research creates them.

## The Bigger Picture

Agriculture faces massive challenges: climate change, labor shortages, food security. AI can help, but only if it's accessible. A smartphone-sized device that a farmer in rural Africa or a greenhouse in Roeselare can use equally well - that's the vision.

This research is one step toward that future. It won't solve everything, but it will prove whether this approach is viable and document what works and what doesn't.

## In One Sentence

**This research explores whether Rust-based machine learning can enable affordable, offline, continuously-learning plant disease detection - filling a gap that current Python-based cloud solutions cannot address.**

---

*Research Project - Semester 5 - Howest MCT*
*Warre Snaet*
