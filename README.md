# Deepfake Detection Project

Multi-architecture deepfake detection implementation using PyTorch.

## Project Structure

```
deepfake_detection/
├── config/             # Configuration files
├── preprocessing/      # Data loading and transforms
├── models/            # Model architectures
├── training/          # Training utilities
├── utils/            # Helper functions
├── checkpoints/      # Saved models
└── logs/             # Training logs
```

## Models

- EfficientNet-B3
- Swin Transformer
- Two-Stream Network
- Xception
- CNN-Transformer Hybrid
- Cross-Attention Hybrid

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Update dataset path in `config/config.py`

3. Train model:

```bash
python main.py --model efficientnet
```

## Training Options

- `--model`: Model architecture to use
- `--resume`: Resume training from checkpoint
- `--test`: Evaluate model on test set

## Dataset Structure

```
dataset/
├── real/             # Real images
└── fake/             # Deepfake images
```
