# Appendix D: Guest Session Report: Digital Fashion and VR Workflows

**Speaker:** Holly, CEO of Studio of New World
**Date:** January 2026
**Student:** Warre Snaet | Howest MCT | Research Project 2025-2026

---

## Speaker Information

| Item | Details |
|:---|:---|
| **Topic** | Digital Fashion and VR Workflows |
| **Organization** | Studio of New World, the first fashion company to operate at industrial scale with a VR-based workflow |
| **Context** | Digital fashion innovation, VR-driven design processes and interdisciplinary team collaboration |

Holly holds a master's degree in Digital Fashion Innovation and is reportedly the first VR fashion designer worldwide to apply this approach at production level. The session focused on how VR can reshape creative workflows, the role of AI in concept generation, and the practical challenges of working in interdisciplinary teams.

---

## 1. The Problem with Digital Fashion

The central problem in the digital fashion industry is poor communication between software packages, especially during the transition from 2D to 3D. Each dimension switch causes a loss of detail and quality, which leads to inefficient workflows and rework. Holly illustrated this by comparing three workflows:

- **Traditional digital workflow:** Concept (3D) → Sketch (2D) → Fashion model (2D) → Real life (3D) / Render (2D). Information is lost at every step.
- **Old VR workflow:** Concept (3D) → Sketch (3D) → Blender (3D) → Render (2D) or 3D-print. A translation to real life remained impossible.
- **Studio New Workflow (new):** Concept (3D) → Sketch (3D) → Fashion (2D/3D) → Real life, render and 3D-print. All output forms are reachable without information loss.

By introducing this new workflow, Holly reduced the production timeline from three weeks to three days.

---

## 2. From 2D Ideas to 3D Decisions

A common mistake is to use VR only as a visualisation tool after decisions have already been made. This leads to slow iteration cycles, expensive revisions and overrefinement of concepts that were weak from the start. When VR is used early in the process, it offers clear advantages: ideas can be tested spatially before a choice is made, decisions become faster and better supported, and problems are identified earlier. The ideal sequence according to Holly is: AI concept → VR modelling → CAD refinement.

---

## 3. Case Studies

Holly illustrated her workflow through three concrete projects:

| Project | Workflow Steps |
|:---|:---|
| **Fortnite** | Collect reference images → use AI for extra references → design in VR → refine in CAD |
| **Erobern Designs** | Briefing → research → texture research → AI enhancement → realisation in VR |
| **Decathlon** | Brief → PDF files → remove logos in Photoshop → VR for sketching → VR modelling |

---

## 4. ChatGPT and Image Prompting

Holly gave practical insights on working with AI image generation and prompting. Important points:

- **Intent and context** are essential in prompting so that the AI does not hallucinate.
- A useful technique is to ask ChatGPT first how it interprets a given image, so you do not have to do that analysis yourself.
- **Seed:** a seed is all the information about the characteristics of something specific, a kind of fingerprint of a style, object or concept. To use seeds effectively, you identify what is worth keeping, store it with context and organise it for reuse.
- AI does 90% of the work, but the remaining 10% (checking for errors such as extra fingers or strange joints) remains essential human work.
- Videos can be generated easily by throwing an image into ChatGPT with an idea and asking it to make a video.

---

## 5. Working in Interdisciplinary Teams

Miscommunication within teams is one of the largest cost drivers in creative projects. Holly addresses this through a fixed approach:

- Compile a content pack as a shared starting point.
- Let AI generate initial design ideas.
- Combine those with the artistic input of the team.
- Work together towards the final design.

An important distinction Holly makes is the following: **VR literacy is not the same as knowing software.** VR literacy is about spatial thinking and decision-making, not about technical software skills. This competence is separate from learning to operate a specific program, and it is exactly this mindset that allows teams to use VR productively.

---

## Personal Reflection

This guest session was relevant for my research project for several reasons:

1. **VR and spatial decision-making.** Although my project does not use VR, the principle of making decisions early with the right tool is directly applicable. In my case, that means choosing the correct deployment target and preprocessing pipeline before investing weeks in model training. Holly's emphasis on early validation maps well to my recommendation in Chapter 5 to deploy to the target device by week 2.

2. **AI as a creative tool, not a replacement.** Holly's 90/10 rule (AI does 90%, humans do the critical 10%) applies to machine learning engineering as well. The model can generate pseudo-labels automatically, but a human still has to validate the confidence thresholds, inspect the failure cases and decide when to stop retraining. The SSL pipeline in this project follows the same division of labour.

3. **Interdisciplinary communication.** Holly's point about miscommunication in teams resonated with the challenges I faced when bridging the Rust backend, the Svelte frontend and the Tauri IPC layer. Each of those layers has its own conventions, and bugs such as the BGRA/RGB mismatch are exactly the kind of silent errors that happen when two disciplines (mobile camera APIs and ML preprocessing) do not communicate clearly.

---

## Key Takeaways

- Switching between 2D and 3D tools causes information loss; a unified 3D workflow avoids this.
- VR should be used early in the decision process, not only as a final visualisation step.
- AI can speed up concept generation dramatically, but human verification remains essential.
- VR literacy is about spatial thinking, not software operation.
- Clear team communication through shared content packs reduces costly misalignment.

---

*Report written: January 2026*
*Student: Warre Snaet*
*Program: MCT, Research Project*
