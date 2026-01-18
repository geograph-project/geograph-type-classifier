## v2.3: Global Distance Dropout & Refined Corruption
**Focus: Breaking Metadata Dependency**

* **Global Dropout Implementation:** Introduced a **30% universal dropout** where the distance is set to `Unknown` (Index 22) for *all* classes. This ensures the `Unknown` tag provides zero predictive value, forcing the model to rely on CLIP visual embeddings.
* **Refined Cross-Far Corruption:** For the remaining 70% of `Cross Far` samples, distance is corrupted to a random short-range value (Indices 0–9).
* **Result:** Promising. Initial testing on the 300-sample set shows the model now successfully discerns between "Far" and "Close" views based on pixels alone, even when both inputs are marked as `Unknown`.
* **Performance:** Achieved an **F1-score of 0.85 for Cross Far** with significantly more "honest" visual classification.

## v2.2: Hard Metadata Blackout (Experimental)
**Focus: Forced Visual Learning**

* **Strategy:** Set `dist_val = 22` (Unknown) for 100% of `Cross Far` training samples.
* **Result:** **Unsuccessful.** Created a "Shortcut Bias" where the model learned that the `Unknown` tag itself was a strong predictor for `Cross Far`. This caused the model to overwhelmingly select `Cross Far` for almost any image lacking distance metadata, including close-range Geographs.

* # v2.1: Stochastic Distance Index Corruption

Resolved Feature Leakage where the model relied exclusively on distance metadata for Cross Far predictions. We implemented a 30% corruption rate on the distance input for Cross Far training samples:

15% forced to Index 22 (Unknown): To improve generalization for the 8M archive images lacking metadata.

15% assigned a Random Index (0–9): Corresponds to distances of 0–256m. This forces the CLIP visual embeddings to override contradictory or erroneous metadata.

This ensures the model identifies "Long Distance Views" based on visual signatures (haziness, scale) rather than just reading the distance field.

Deterministic Corruption: Implemented a row-level strategy by seeding the local random generator with the gridimage_id. This ensures 100% reproducibility across training runs without affecting global random states.

* **Result:** **Unsuccessful.** was found to not notablely impve the detection of Cross-Far (when distance unknown or wrong), althoug seems to have improved other classes a bit, as less reliant on distance

# v2: Dataset Expansion & Class Balancing

Focus: Scale and Bias Reduction

Increased Sample Size: Expanded the dataset to approximately 430,000 samples.

Bias Mitigation: Attempted to balance input classes to minimize the heavy bias toward Geograph images (which previously made up 83% of the samples).

Enhanced "From Drone" Training: Significantly increased the sample count for drone-perspective images by trawling the archive for visual matches (including "drone-like" views that aren't technically drones) to improve the model's recognition of high vantage points.

Script Status: Model architecture and training script remained unchanged from v1.

# v1: Initial Scale-Up

Focus: Viability Testing

Baseline Dataset: Increased to 130,000 samples, moving beyond the initial v0 prototype (1k samples).

Objective: Designed to provide a statistically significant sample size for testing architectural performance without the overhead of the full archive.
