# MNIST Digit Classification using Custom HOG Features

## Overview
This project aims to classify handwritten digits from the MNIST dataset using a traditional computer vision approach.  
Instead of deep learning, it leverages handcrafted feature extraction using the **Histogram of Oriented Gradients (HOG)** method, implemented entirely from scratch using the **Roberts Cross operator** for edge detection.  
The extracted HOG vectors are used to train classical machine learning models including **XGBoost**, **Random Forest**, and **SVM**, achieving competitive performance.

---

## Features

**Custom HOG Feature Extraction:**
- Implemented from scratch without relying on external libraries like OpenCV or scikit-image.
- Uses **Roberts Cross operator** for computing gradient magnitude and orientation.
- Soft-binned histograms with bilinear interpolation across 8 angle bins.
- Block-wise normalization using L2-norm for spatial invariance.

**Visualization Tools:**
- Display of **gradient magnitudes** to understand edge intensity.
- Overlay of **pixel-wise orientation arrows** to visualize gradient directions.

**Model Training & Evaluation:**
- Trained classical ML models (XGBoost, Random Forest, and SVM) on extracted HOG features.
- Evaluated performance on a held-out test set from MNIST using accuracy metrics.

---

## Dataset

**Source:** MNIST Handwritten Digit Dataset (from `tf.keras.datasets.mnist`)  
**Size:** 10,000 grayscale digit images (reduced from 60,000 for fast experimentation)  
**Image Format:** 28×28 grayscale  
**Classes:** 10 (digits 0–9)

---

## Data Preprocessing

**HOG Feature Extraction:**
- For each 28×28 image:
  - Compute gradient magnitude and orientation using **Roberts Cross**.
  - Divide into 2×2 spatial blocks.
  - Construct 8-bin orientation histograms per pixel and normalize over blocks.

**Train-Test Split:**
- 80-20 split for training and evaluation using `train_test_split`.

**Final Feature Shape:**
- Each image is represented by a **1D HOG feature vector** after flattening and normalization.

---

## Model Architectures

**1. XGBoost Classifier**
- `n_estimators=200`  
- `max_depth=5`  
- `learning_rate=0.1`  
- `tree_method='hist'`  
- Accuracy: **~92.85%**

**2. Random Forest Classifier**
- `n_estimators=250`  
- Accuracy: **~91.4%**

**3. Support Vector Machine (SVM)**
- `kernel='rbf'`  
- `C=1.0`  
- Accuracy: **~94.65%**

---

## Visualizations

**1. Gradient Magnitude**
```python
visualize_gradient_magnitude(mag)
```
Displays the **gradient magnitude** image, highlighting edges and intensity changes across the digit.  
Brighter areas indicate stronger gradients, revealing where most edge activity occurs in the input image.

**2. Pixel-wise Gradient Orientation**
```python
visualize_gradients_within_pixels(image, mag, direction)
```
Overlays **red arrows** on the original image to indicate the **direction** and **relative strength** of gradients at each pixel.  
This helps visually understand the orientation features that feed into the HOG descriptor.

These visualizations are helpful for:
- Understanding edge strength and orientation at a pixel level  
- Verifying the correctness of the HOG feature extraction  
- Interpreting what kind of features the model learns from
