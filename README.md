# FrameVision: Multimodal Sentiment Analysis System

<div align="center">

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Local Training](#local-training)
  - [SageMaker Training](#sagemaker-training)
  - [Inference](#inference)
  - [Deployment](#deployment)
- [Technical Details](#technical-details)
- [License](#license)

## 🔍 Overview
FrameVision is a state-of-the-art multimodal sentiment analysis system that processes video, audio, and text data simultaneously to predict emotions and sentiments. Built with PyTorch and deployable on AWS SageMaker, it provides both local and cloud-based training options.

## ✨ Features
- Multimodal analysis combining video, audio, and text
- Support for both local and cloud-based training
- AWS SageMaker integration for scalable deployment
- Real-time inference capabilities
- Comprehensive evaluation metrics
- Modular architecture for easy extensions

## 📊 Dataset
This project uses the [MELD (Multimodal EmotionLines Dataset)](https://github.com/SenticNet/MELD) which contains:
- 13,000 utterances from Friends TV series
- 7 emotion categories (anger, disgust, fear, joy, neutral, sadness, surprise)
- 3 sentiment classes (positive, negative, neutral)
- Multimodal features (text, audio, and video)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FrameVision.git
cd FrameVision

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install training dependencies
pip install -r training/requirements.txt

# Install deployment dependencies (if needed)
pip install -r deployment/requirements.txt

# Install FFmpeg
python training/install_ffmpeg.py
```

## 📁 Project Structure
```
FrameVision/
├── training/
│   ├── model.py              # Model architecture
│   ├── meld_dataset.py       # Dataset loader
│   ├── train.py              # Local training script
│   └── train_sagemaker.py    # SageMaker training script
├── deployment/
│   ├── deploy_endpoint.py    # SageMaker deployment
│   └── inference.py          # Inference pipeline
└── requirements.txt          # Dependencies
```

## 💻 Usage

### Local Training
```bash
python training/train.py \
    --epochs 20 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --data-path /path/to/meld/dataset
```

### SageMaker Training
```bash
python training/train_sagemaker.py \
    --bucket-name your-bucket \
    --role-arn your-role-arn
```

### Inference
```bash
python deployment/inference.py \
    --video-path /path/to/video.mp4 \
    --model-path /path/to/model.pth
```

### Deployment
```bash
python deployment/deploy_endpoint.py \
    --model-name framevision-model \
    --instance-type ml.p3.2xlarge
```

## 🔧 Technical Details

### Model Architecture
- **Video**: ResNet-based frame encoder (224×224 input, max 30 frames)
- **Audio**: MEL spectrogram features (64 bands)
- **Text**: BERT embeddings with linear projection
- **Fusion**: Multi-head attention mechanism


## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
Made with ❤️ by Parth Soni
</div>
