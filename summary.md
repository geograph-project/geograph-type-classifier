# Predicting Geograph Type Tags via CLIP & Spatial Metadata

## Abstract
This project presents a machine learning ecosystem for the automated classification of Geograph images into their native “Type” taxonomies. Adapting the methodology introduced by the **CLIP the Landscape** project, our approach utilizes a multi-modal feature fusion of precomputed **CLIP ViT-B/32** image embeddings and quantized spatial metadata. 

While previous work focused on "Context Tags," this implementation extends the framework to predict **7 distinct "Type" tags**, addressing the unique challenges of archival categories that often include non-visual or administrative attributes. We implement a Multi-Layer Perceptron (MLP) architecture with **Stochastic dropout** to mitigate bias and improve generalization across a newly curated dataset of approximately 430,000 samples. We provide the complete training pipeline, precomputed embeddings, and a production-ready model to facilitate enhanced spatial discovery and digital heritage management.

---

## Inspiration & Acknowledgements
This project was directly inspired by the architectural approach and core concepts of the **CLIP the Landscape** project. 
https://doi.org/10.1016/j.rsase.2025.101824

While *CLIP the Landscape* demonstrated the efficacy of fusing CLIP embeddings with spatial metadata for "Context Tags," this repository adapts and extends that methodology specifically for the Geograph **"Type"** tag system. We have expanded the dataset and implemented a custom classification head to account for the specific nuances of archival type-tagging.

---

## Classification Categories
The model is trained to categorize images into seven core Geograph "Type" tags. These categories capture a mix of perspective, proximity, and administrative classification:

* **Aerial**: High-altitude perspectives (traditionally from planes/helicopters).
* **Close Look**: Macro or tight-framed shots where the subject is in immediate proximity.
* **Cross Far**: Identifies long-range "Cross Grid" images. While our standard "Cross Grid" tag can be inferred from coordinates, "Cross Far" acts as a proxy for images that are visually long-range (e.g., a subject significantly far across a grid line).
* **Extra**: Supplementary or outlier imagery within the dataset.
* **Geograph**: Standard archival imagery matching the core project criteria.
* **Inside**: Interior shots or enclosed environments.
* **From Drone**: *Experimental.* An attempt to classify modern drone imagery uniquely. Note: This is not a (yet) a formal Geograph tag, and current model performance is limited due to a smaller training set relative to other classes. Generally captures imagery at a closer range than traditional Aerial imagery.

---

## Technical Architecture & Pipeline

The model utilizes a multi-modal fusion approach, combining high-level visual feature extraction with spatial metadata to predict archival classifications.



### **The Pipeline Process**

1.  **Visual Feature Extraction:** Raw image pixels are processed through the **CLIP ViT-B/32** encoder to generate a 512-dimensional vector representing the semantic content of the scene.
2.  **Spatial Metadata Integration:** We incorporate geographic context by calculating the **Euclidean distance** ($d$) between the camera coordinates ($C$) and the subject coordinates ($S$):
    $$d = \sqrt{(x_s - x_c)^2 + (y_s - y_c)^2}$$
    This distance scalar is quantized to provide the model with a sense of scale and proximity.
3.  **Feature Fusion:** The 512-D visual embedding and the distance metadata are concatenated into a single input vector.
4.  **Classification Head:** This fused vector is passed into a **Multi-Layer Perceptron (MLP)**. To ensure the model does not over-fit on specific visual patterns—especially for tags that represent administrative rather than purely visual data—we employ **Stochastic Dropout** during the training phase.
5.  **Output:** The model produces probability scores across the 7 Geograph "Type" tags.

---

### **Datasets & Resources**
* **Curated Dataset:** ~430k samples with precomputed CLIP embeddings and metadata. [Available on Kaggle](https://www.kaggle.com/datasets/barrybhunter/geograph-types-dataset-1)
* **Training Pipeline:** PyTorch implementation optimized for CPU/GPU. [View on GitHub](https://github.com/geograph-project/geograph-type-classifier/)
