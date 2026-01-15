# geograph-type-classifier
Building a ML Model to predict Geograph Type Tags

# Overview

This repository will contain three things
1. Access to prepared datasets for training. (multiple sizes) 
2. A torch implementation of a training pipeline.
3. A functional model prepared with above, usable for immidate inference. Demo script provided. 

Although the by providing the raw dataset, we hope others can try building an even better model. 


# Dataset Implementation & Training Guide

This dataset is designed for multimodal classification using CLIP embeddings and distance-based spatial metadata. To achieve the best results, please follow these implementation strategies.

### 1. Loss Weighting & Class Disambiguation
* **Weighted Loss:** It is highly recommended to incorporate the provided `weight` column into your loss function (e.g., using the `weight` parameter in `BCEWithLogitsLoss`). These weights mitigate label uncertainty and ensure the model prioritizes high-confidence samples.
* **The 'Cross Far' Strategy:** We suggest mapping the raw **"Cross Grid"** label to **"Cross Far"** during the training phase.
    * **Objective:** This encourages the model to learn the specific visual "texture" and atmospheric perspective of long-distance shots.
    * **Logic:** While "Cross Near" images (those that span a gridline at close range) are best handled via coordinate geometry, this model is intended to identify images that *visually* represent long-range perspectives, regardless of coordinate precision.

### 2. Pseudo-Labeling for Sparse Classes ('From Drone')
* **The 'From Above' Proxy:** To address the scarcity of authentic drone data, we have included the **"From Above"** tag. These represent high-vantage, low-level aerial perspectives (e.g., shots from bridges, towers, or cliffs) that are visually indistinguishable from drone-captured media.
* **Training Instruction:** Map all **"From Above"** instances to the **"From Drone"** class during training to provide the model with a sufficient learning signal for the "downward-looking aerial" concept.
* **Visual Constraints:** Training samples for this class should ideally lack immediate foreground elements (like railings or window frames) to ensure the model learns the perspective itself rather than the structure the photographer is standing on.
* **Data Integrity:** These remain distinct tags in the raw metadata, allowing researchers to evaluate performance on "True Drone" vs. "Pseudo-Drone" samples during testing.

### 3. Recommended Class Schema
For optimal performance and to minimize semantic overlap, use the following classification targets:

```python
CLASSES = [
    "Aerial", 
    "Close Look", 
    "Cross Far", 
    "Extra", 
    "Geograph", 
    "Inside", 
    "From Drone"
]
```

# Pre-Trained Model

## Technical Requirements

* **Embeddings:** The model is built on **CLIP ViT-B/32** image embeddings (512-dimensional vectors).
* **Metadata:** Input requires a quantized **approximate distance** index (0-22), representing distance buckets from the subject.
* **Inference:** You can use the pre-computed embeddings provided in the dataset or generate them from raw images using the OpenAI CLIP library.

## Quick Start: Using the Model

```python
from model import GeographModel

# Load the latest best-performing weights
model, checkpoint = GeographModel.load_checkpoint("best_geograph_model.pth")

# Predict on a single image embedding
# vec: CLIP embedding (dim=512)
# dist: Distance bucket index (0-22)
logits = model(vec, dist)
probabilities = logits.sigmoid()

for i, class_name in enumerate(checkpoint['classes']):
    print(f"{class_name}: {probabilities[0][i]:.2%}")

#Reminder, the model produces the "Cross Far" class, which is intended to indicate a long distance shot, so treat as "Cross Grid", short range cross-grids can be detected from coordiates. 
