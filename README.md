# Dental Caries Detection and Segmentation using U-Net

## Overview

An AI-based dental caries segmentation system that analyzes panoramic dental X-ray images using a U-Net deep learning model.

## Features

- Dental caries segmentation
- X-ray and segmentation visualization
- Predicted caries area calculation
- Detected region analysis
- Automated dental caries report
- Streamlit web application

## Model

- Architecture: U-Net
- Input: Grayscale panoramic dental X-ray
- Input resolution: 768 x 768
- Task: Binary image segmentation
- Decision threshold: 0.80

## Dataset Split

- Training images: 70
- Validation images: 15
- Test images: 15
- Training patches: 350
- Validation patches: 75
- Test patches: 75

## Test Performance

| Metric | Score |
|---|---:|
| Precision | 92.57% |
| Recall | 80.62% |
| F1 / Dice | 86.18% |
| IoU | 75.72% |

## Technologies

Python, PyTorch, U-Net, OpenCV, NumPy, Pillow, Matplotlib and Streamlit.

## Pipeline

Dental X-ray -> Preprocessing -> U-Net -> Probability Map -> Thresholding -> Caries Mask -> Area Calculation -> Report

## Run the Application

Install dependencies:

pip install -r requirements.txt

Run:

streamlit run app.py

## Disclaimer

This project is intended for research and educational purposes. AI results should not be considered a medical diagnosis and should be reviewed by a qualified dental professional.