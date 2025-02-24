# FrameVision Model Training & Inference

This repository contains code for a multimodal sentiment analysis workflow using text, video, and audio data. It uses the [MELD dataset](https://github.com/SenticNet/MELD) for training and evaluation.

## Overview
- **model.py / MultimodalSentimentModel**: Defines a PyTorch model that encodes text, video, and audio features.  
- **meld_dataset.py**: Loads MELD CSV files and associated `.mp4` video clips, extracting frames and audio.  
- **train.py**: Trains the model locally with PyTorch, logging metrics and saving the best model.  
- **deploy_endpoint.py**: Demonstrates how to deploy the model on SageMaker.  
- **inference.py**: Provides an inference entry point that runs video, audio, and text through the trained model.  
- **train_sagemaker.py**: Sets up a SageMaker PyTorch estimator for remote training.  
- **install_ffmpeg.py**: Installs FFmpeg for audio extraction.

## Installation
1. Clone this repository and open the folder.  
2. Install dependencies for training or inference:
   ```bash
   pip install -r training/requirements.txt
   ```
   For deployment dependencies:
   ```bash
   pip install -r deployment/requirements.txt
   ```
3. Ensure FFmpeg is installed or execute:
   ```bash
   python training/install_ffmpeg.py
   ```

## Training Locally
1. Place your MELD CSV files and corresponding `.mp4` videos in the designated folders.  
2. Adjust paths in `train.py` if needed (or pass them as arguments).  
3. Run:
   ```bash
   python training/train.py --epochs 20 --batch-size 16
   ```

## Training on SageMaker
1. Create an Amazon S3 bucket and upload train/validation/test data.  
2. Modify `train_sagemaker.py` for S3 paths and IAM role.  
3. Run:
   ```bash
   python train_sagemaker.py
   ```

## Inference
1. Place the trained model `.pth` file into the expected path (by default in the `deployment` folder).  
2. Check the model path in `inference.py` and run:
   ```bash
   python deployment/inference.py
   ```
   This script downloads a video from S3, processes frames/audio, applies Whisper for transcription, and classifies emotional/sentiment labels.

## Deployment
To set up a SageMaker endpoint:
```bash
python deployment/deploy_endpoint.py
```
This instantiates the endpoint using the model files and inference script.

## Notes
- Video frames are resized to 224×224 and stacked up to 30 frames.  
- Audio features use a 64-mel-spectrogram (via `torchaudio`).  
- Text embeddings come from a frozen BERT backbone with a linear projection.  
- MELD includes emotion/sentiment labels, which are predicted jointly.

## License
Check the dataset’s license and associated usage terms. This project is provided for educational or testing purposes.
