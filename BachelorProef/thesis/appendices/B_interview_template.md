# Appendix B: Interview Question Template

These questions are meant for the external reflection interviews (Chapter 4). They should be adapted to the expert's background and area of expertise.

---

## General Information

- **Date of interview:** ___
- **Name of interviewee:** ___
- **Organization / company:** ___
- **Role / title:** ___
- **Relevant expertise:** ___
- **Format:** (in person / video call / email exchange)

---

## Section 1: Context and Problem Relevance

1. In your experience, how significant is the problem of plant disease detection in agricultural practice? Is early detection something farmers actively seek solutions for?

2. How well do current solutions (laboratory analysis, cloud-based apps) actually perform in practice? What are the main pain points you tend to see?

3. What is your opinion on the requirement for offline, on-device operation? Is internet connectivity a genuine barrier in the agricultural contexts you work with?

## Section 2: Technical Approach

4. We use a semi-supervised learning approach (pseudo-labeling with a 90% confidence threshold) to bring the labeled data requirement down from 100% to 20%. How do you evaluate this approach? Do you see risks of error propagation?

5. The entire system is implemented in Rust using the Burn ML framework and compiles into a binary of roughly 26 MB. What is your reaction to this technology choice compared to the standard Python/PyTorch stack?

6. Our benchmarks show 0.39 ms inference on a desktop GPU and 80 ms on an iPhone 12. Are those latency numbers sufficient for the use cases you have in mind?

7. We chose to deploy on consumer devices (phones, laptops) rather than on dedicated edge hardware (Jetson), based on cost-performance benchmarks. Would you agree with that decision?

## Section 3: Incremental Learning and Scalability

8. Our experiments show that adding a new class to a 30-class model causes 6× more forgetting than adding one to a 5-class model. In your experience, how often do production models need to accommodate new classes? Is this a relevant concern for you?

9. Which methods do you use (or recommend) for mitigating catastrophic forgetting when deployed models are extended?

10. Do you see a practical path from 38 classes (PlantVillage) to coverage of the diseases that are relevant to your region or crop portfolio?

## Section 4: Deployment and Real-World Viability

11. What do you consider the biggest barriers to deploying an AI-based diagnostic tool to farmers in practice?

12. How would you distribute and update an offline application for end users who may have limited digital literacy?

13. What level of accuracy would you consider acceptable for a field-deployed plant disease detection tool? Is 85% sufficient, or does the threshold need to be higher?

## Section 5: Broader Impact

14. What economic value do you see in reducing labeling costs and in eliminating cloud inference charges?

15. Do you see societal or environmental benefits in making plant disease detection more accessible to smallholder farmers?

16. What future developments or research directions would you recommend for this type of project?

---

## Closing

17. Is there anything else you would like to add that we have not covered?

18. May I reference you by name and organization in the thesis?

---

*Note: The complete transcripts or written responses from each interview are included alongside this template in the final thesis submission.*
