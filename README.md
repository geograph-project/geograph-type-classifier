# geograph-type-classifier
### Predicting Geograph Type Tags via CLIP & Spatial Metadata

## Overview

This repository provides a complete machine learning ecosystem for categorizing Geograph images into their native "Type" tags. The project bridges raw visual data with geographic context to automate archival classification.

The project consists of three core components:

1. **Curated Datasets:** Access to prepared training sets featuring precomputed **CLIP ViT-B/32** image embeddings (512-dimensional), quantized distance metadata, and ground-truth Type tags.
   - *Current Status:* ~130k samples. Published to Kaggle: https://www.kaggle.com/datasets/barrybhunter/geograph-types-dataset-1
   - *Target:* 5.0M samples.
2. **Training Pipeline:** A robust PyTorch implementation designed for efficiency on both CPU and GPU. The pipeline is fully functional within Google Colab environments for immediate experimentation.
3. **Production-Ready Model:** A pre-trained functional model using the schema above. We provide a demo script for immediate inference on new image/metadata pairs.

By providing the raw datasets and embeddings, we hope to enable the community to build even more sophisticated models for geographic visual understanding.

---
*Note: This project was "Vibe Coded" in collaboration with Gemini 3 (Flash).*

## Credits & Citations

The architectural approach and the core concept of fusing **CLIP Embeddings** with **Spatial Metadata** were directly inspired by the **CLIP the Landscape** project. While that project focused on Geograph "Context Tags," this implementation adapts the methodology specifically for predicting Geograph **"Type"** tags using an expanded dataset and a custom classification head.

If you find this work or the underlying methodology useful, please cite the original research paper:

> **Ilya Ilyankou, Natchapon Jongwiriyanurak, Tao Cheng, James Haworth,** > *CLIP the landscape: Automated tagging of crowdsourced landscape images*,  
> **Remote Sensing Applications: Society and Environment**, Volume 41, 2026, 101824, ISSN 2352-9385.  
> [https://doi.org/10.1016/j.rsase.2025.101824](https://doi.org/10.1016/j.rsase.2025.101824)

**SpaceTimeLab Implementation:** [github.com/SpaceTimeLab/ClipTheLandscape](https://github.com/SpaceTimeLab/ClipTheLandscape)

## Technical Requirements & Implementation

### 1. Embeddings & Distance Logic
* **Embeddings:** The model utilizes **CLIP ViT-B/32** image embeddings (512-dimensional vectors). If generating your own, ensure you use the ViT-B/32 variant.
* **Distance Quantization:** The model relies on a quantized power-of-two distance index. This helps distinguish between "Cross Near" and "Cross Far" perspectives.

**Distance Calculation (Python):**
```python
import math

def calculate_geograph_distance(e1, n1, e2, n2):
    """Calculates Euclidean distance and snaps to floor power-of-two."""
    if not (n1 > 0 and e2 > 0): return "Unknown"
    dist = math.sqrt((e1 - e2)**2 + (n1 - n2)**2)
    if dist == 0: return "0"
    return str(int(2**math.floor(math.log2(dist))))
```

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
Code below demonstrates converting the tags in the dataset into this schema.

## Distance Quantization Logic

The model uses a specific logarithmic binning for distance metadata. We provide a `get_dist_idx()` helper to convert these values into model inputs.

### 1. Distance Mapping
The model expects an integer index based on the following power-of-two buckets:
* **0**: Exact location / 0m
* **1-21**: Logarithmic buckets (2^0m to 2^20m)
* **22**: Unknown / Missing data

### 2. Manual Calculation
If you are working with raw National Grid (eg OSGB36 or Irish Grid) coordinates, use this logic to match the training data:

```python
import math

def get_geograph_distance(e1, n1, e2, n2):
    if not (n1 > 0 and e2 > 0):
        return "Unknown"
    # Euclidean distance
    d = math.sqrt((e1 - e2)**2 + (n1 - n2)**2)
    if d == 0: return "0"
    # Snap to nearest floor power of 2
    return str(int(2**math.floor(math.log2(d))))

# Convert coordinate distance to model index
raw_dist = get_geograph_distance(nateast, natnorth, vpeast, vpnorth)
dist_input = get_dist_idx(raw_dist)
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
```

# Suggested Training Setup (PyTorch)
When training, use the From Above tag to "boost" the From Drone neuron:

```python
CLASSES = ["Aerial", "Close Look", "Cross Far", "Extra", "Geograph", "Inside", "From Drone"]
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

# Generate power-of-2 map dynamically up to 2^20 (1,048,576)
DIST_MAP = {str(2**i): i + 1 for i in range(21)}
DIST_MAP["0"] = 0
DIST_MAP["Unknown"] = 22
DIST_MAP[""] = 0 #empty string would imply zero distance not unknown

def get_dist_idx(val):
    val_str = str(val).strip()
    return DIST_MAP.get(val_str, DIST_MAP["Unknown"])

CLIP_DIM = 512 # Standard CLIP embedding dimension

# Define a function to decode the Base64-encoded vector
def decode_vector(encoded_str):
    try:
        binary_data = base64.b64decode(encoded_str)
        return np.frombuffer(binary_data, dtype=np.float32)
    except Exception as e:
        # Catch any other unpacking errors
        print(f"Warning: Error unpacking binary data: {e}. Returning zero vector.")
        return [0.0] * CLIP_DIM

class MultimodalGeographDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Process Embeddings
        vec = torch.tensor(decode_vector(item['embeddings']), dtype=torch.float32)

        # Process Distance
        dist_idx = torch.tensor(get_dist_idx(item['distance']), dtype=torch.long)

        # Process Multi-labels (One-Hot Encoding)
        label_tensor = torch.zeros(len(CLASSES))
        raw_types = item['types'].split(',')
        for t in raw_types:
            # Note: We treat "Cross Grid" as "Cross Far" for the target label if dist > 256
            clean_t = t.strip()
            if clean_t == "Cross Grid":
                if str(item['distance']).isdigit() and int(item['distance']) > 256:
                    clean_t = "Cross Far"
                elif len(raw_types) == 1: ##if was ONLY gross grid, then change it
                    clean_t = "Geograph" # Visual proxy
                else: #else just ignore the CR (can still be inside etc)
                    continue

            # Funnel 'From Above' into the Drone class
            # its a fake tag for images that look like drone, but probably
            # arent. but included so the model can learn from them.
            if clean_t == "From Above":
                clean_t = "From Drone"

            if clean_t in CLASS_TO_IDX:
                label_tensor[CLASS_TO_IDX[clean_t]] = 1.0

        # Calculate Weight
        weight = torch.tensor(item.get('weight', 1.0), dtype=torch.float32)

        return vec, dist_idx, label_tensor, weight
```

See the training notebook, for more complete implementation.
