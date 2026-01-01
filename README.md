# COMP3105-CNN-Image-Classification
Assignment 4: CNN for Domain-Specific Classification

Implementation of a Convolutional Neural Network for domain-specific image classification using PyTorch.

## Architecture
- 2-layer CNN with Conv2d → ReLU → MaxPool2d blocks
- Fully connected classifier with 256 hidden units
- Cross-entropy loss with Adam optimizer

## Strategy
- Trained exclusively on in-domain data
- Designed to maintain poor performance on out-domain data
- Achieved ~52% in-domain accuracy while meeting constraints

**Topics covered:**
- Convolutional Neural Network implementation
- PyTorch framework usage
- Dataset loading and preprocessing
- Domain specialization strategy

**Files:**
- `A4codes.py` - CNN implementation in PyTorch
- `A4Report.pdf` - Architecture design and analysis
- `requirements.txt` - Deep learning dependencies

## Quick Start
```bash
pip install -r Assignment4/requirements.txt

## Technologies
Python 3.10

NumPy, SciPy

PyTorch, torchvision

Matplotlib (for visualization)
