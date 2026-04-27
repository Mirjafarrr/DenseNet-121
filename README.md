# DenseNet-121 Image Classification on CIFAR-10

<div>![DenseNet-121 Architecture]('DenseNet-121 Diagram.png')</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

A modular, from-scratch implementation of the **DenseNet-121** architecture in PyTorch, trained and evaluated on the **CIFAR-10** benchmark dataset. Images are upscaled from 32×32 to 224×224 to fully leverage DenseNet's deep feature extraction pipeline.

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training](#training)
  - [Inference](#inference)
- [License](#-license)

---

## 🔍 Overview

This project provides a clean, modular PyTorch implementation of **DenseNet-121** (Densely Connected Convolutional Networks) applied to the CIFAR-10 image classification task. The key design decisions are:

---

## 📊 Dataset

The model is trained on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of **60,000** color images across **10 balanced classes**.

| Property | Details |
|---|---|
| Total images | 60,000 |
| Image resolution (raw) | 32 × 32 |
| Image resolution (model input) | 224 × 224 |
| Classes | 10 |
| Images per class | 6,000 |
| Train / Validation split | 50,000 / 10,000 |

**Classes:** `airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---

## 🏗️ Model Architecture

DenseNet (Densely Connected Convolutional Networks) introduces **dense connectivity**: every layer receives feature maps from *all* preceding layers and passes its own feature maps to *all* subsequent layers within the same dense block. This design has three major benefits:

- **Alleviates the vanishing-gradient problem** by providing shorter gradient paths back to earlier layers.
- **Strengthens feature propagation** by reusing learned representations throughout the network.
- **Reduces parameter count** compared to equivalent ResNets, since feature reuse avoids redundant learning.

### DenseNet-121 Architecture Breakdown

| Stage | Component | Configuration |
|---|---|---|
| **Stem** | Convolution | 7×7, stride 2, padding 3 → 64 channels |
| **Stem** | BatchNorm + ReLU + MaxPool | 3×3, stride 2 |
| **Dense Block 1** | 6 dense layers | Growth rate k = 32 |
| **Transition 1** | 1×1 Conv + 2×2 AvgPool | Compression θ = 0.5 |
| **Dense Block 2** | 12 dense layers | Growth rate k = 32 |
| **Transition 2** | 1×1 Conv + 2×2 AvgPool | Compression θ = 0.5 |
| **Dense Block 3** | 24 dense layers | Growth rate k = 32 |
| **Transition 3** | 1×1 Conv + 2×2 AvgPool | Compression θ = 0.5 |
| **Dense Block 4** | 16 dense layers | Growth rate k = 32 |
| **Classification Head** | Global AvgPool + Linear | 10 output classes |

> Each **dense layer** internally follows a Bottleneck design: **BN → ReLU → 1×1 Conv → BN → ReLU → 3×3 Conv**, expanding to 4k channels before projecting back to k feature maps.

---

## 📂 Project Structure

```
DenseNet-121/
│
├── data/               # CIFAR-10 dataset files (git-ignored)
├── models/             # Saved model weights (.pth)
├── src/
│   ├── model.py        # DenseNet-121 architecture definition
│   ├── data_setup.py   # Dataset loading and transforms
│   └── train.py        # Training and validation loops
├── predict.py          # Inference script for real-world testing
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

---

## ⚙️ Installation

**Prerequisites:** Python 3.8+, pip, and a CUDA-capable GPU (optional but recommended).

```bash
# 1. Clone the repository
git clone https://github.com/Mirjafarrr/DenseNet-121.git
cd DenseNet-121

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Training

Configure hyperparameters directly at the top of `train.py`, then launch:

```bash
python train.py
```

The training loop will:
1. Automatically download CIFAR-10 if not already present.
2. Train for the configured number of epochs, printing loss and accuracy each epoch.
3. Save the best-performing model checkpoint to `models/best_densenet.pth` based on validation loss.
4. Plot training and validation loss/accuracy curves via `plot_training_results`.


### Inference

Run `predict.py` to classify any publicly accessible image from a URL:

```bash
python predict.py
```

Or call `predict_image()` directly in your own script:

```python
from predict import predict_image

predict_image("https://example.com/your-image.jpg")
```

**Example output:**

```
--- Prediction Result ---
Predicted Class : DEER
Confidence      : 97.43%
--------------------------
```

The script will:
1. Load the saved weights from `models/best_densenet.pth`.
2. Fetch and preprocess the image from the provided URL.
3. Run a forward pass with `torch.no_grad()` for efficient inference.
4. Print the predicted class and softmax confidence score.

---


## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*Built with PyTorch · CIFAR-10 · DenseNet*

</div>