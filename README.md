# 🚀 HFA-YOLO: Enhancing Asset Visibility in Smart Energy Systems

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **Enhancing Asset Visibility in Smart Energy Systems: A Computational Intelligence Framework for Heterogeneous Infrastructure Inspection**
> 
> *Authors: Li Tan, Yibo Li, Dongjiao Ge, Xiaofeng Lian, Yixin Peng*

This repository contains the official PyTorch implementation of **HFA-YOLO**, a lightweight and highly efficient object detection framework based on YOLOv11, specifically tailored for heterogeneous energy infrastructure inspection (e.g., Photovoltaics, Transmission Towers, Wind Turbines) in UAV aerial imagery.

## 📖 Abstract

The fragmentation of energy infrastructure creates an Asset Visibility Gap. While UAV imagery bridges this gap, precise perception faces challenges from **feature dissipation** of tiny assets and **boundary aliasing** in dense arrays. To address this, we propose **HFA-YOLO**. With only **3.06M parameters**, our method achieves an optimal accuracy-efficiency trade-off, realizing significant mAP50 and FPS gains across heterogeneous energy scenarios compared to the state-of-the-art baselines.

## ✨ Key Features

* **Space-to-Depth Convolution (SPD-Conv):** Establishes a lossless down-sampling channel to prevent fine-grained feature vanishing for tiny targets.
* **Detail-Enhanced Unit (DEU):** Leverages explicit gradient priors to sharpen boundaries, effectively resolving the boundary adhesion issue in dense object arrays (e.g., solar panels).
* **Hybrid Feature Aggregation (HFA):** A frequency-spatial collaborative filtering module that enhances weak features and robustly suppresses high-frequency industrial environmental noise (e.g., specular reflections).

## 📊 Main Results

Generalization Validation and Efficiency Comparison on Heterogeneous Datasets:

| Dataset | Model | Precision (%) | Recall (%) | mAP50 (%) | mAP50:95 (%) | End-to-End FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Photovoltaics** | YOLOv11n | 80.48 | 76.41 | 83.61 | 64.89 | **865.49** |
| | **HFA-YOLO-N (Ours)** | **83.05** | **82.09** | **85.73** | **67.31** | 592.91 |
| **Transmission Tower**| YOLOv11n | 76.90 | 73.24 | 73.82 | 42.17 | 105.07 |
| | **HFA-YOLO-N (Ours)** | **82.88** | **75.20** | **78.73** | **43.23** | **105.92** |
| **Wind Turbine** | YOLOv11n | 82.33 | 82.28 | 87.39 | 66.46 | **888.30** |
| | **HFA-YOLO-N (Ours)** | **84.10** | **84.76** | **88.79** | **67.08** | 623.45 |
| **VisDrone2019** | YOLOv11n | 43.70 | 33.60 | 33.10 | 19.20 | 236.41 |
| | **HFA-YOLO-N (Ours)** | **49.15** | **36.51** | **37.28** | **22.26** | **350.88** |

*(Tested on an NVIDIA RTX 4070S GPU)*
