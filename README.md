# Deepfake Detection Project

Multi-architecture deepfake detection implementation using PyTorch.

## Project Structure

```
deepfake_detection/
├── config/             # Configuration files
├── preprocessing/      # Data loading and transforms
├── models/            # Model architectures
│   ├── efficientnet.py
│   ├── swin.py
│   ├── two_stream.py
│   ├── xception.py
│   ├── cnn_transformer.py
│   └── cross_attention.py
├── training/          # Training utilities
├── utils/
│   └── visualization.py  # Plotting and visualization tools
├── checkpoints/      # Saved models
└── logs/             # Training logs
```

## Requirements

- Python 3.7+
- PyTorch 1.9.0+
- CUDA (optional, will run on CPU if not available)

### Core Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
efficientnet-pytorch>=0.7.1
timm>=0.5.4
numpy>=1.19.5
Pillow>=8.3.1
scikit-learn>=0.24.2
tqdm>=4.62.2
```

### Visualization Dependencies

```
matplotlib>=3.4.3
seaborn>=0.11.2
pandas>=1.3.0
tensorboard>=2.7.0
```

### Additional Dependencies

```
opencv-python>=4.5.3
albumentations>=1.1.0
```

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

## Hardware Support

The project automatically adapts to available hardware:

- If CUDA-capable GPU is available: Uses GPU for training (recommended)
- If no GPU is available: Falls back to CPU training
- Multi-GPU: Automatically utilizes multiple GPUs if available

### CPU Training

When running on CPU:

- Batch size might need adjustment (try smaller batches)
- Training will be slower but all functionality remains
- Modify batch size in `config.py`:

```python
# For CPU training, use smaller batch size
Config.BATCH_SIZE = 16  # Adjust based on memory
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

## Visualizations

The training process automatically generates and saves the following plots:

### Training Progress

- Loss history over epochs
- Accuracy history over epochs
- Learning rate changes
- Training vs Validation metrics

### Model Performance

- Confusion matrix
- ROC curve with AUC score
- Prediction probability distribution
- Summary of key metrics (Accuracy, AUC-ROC, F1-Score)

All plots are saved in the working directory under `plots/`.

## Checkpoints

- Checkpoints are automatically saved after each epoch
- Best model and latest model are preserved
- Checkpoints are automatically zipped for easy transfer
- Structure: `checkpoints/{model_name}/[last.pth|best.pth]`

## Kaggle Integration

To run on Kaggle:

1. Create new notebook with GPU
2. Clone repository
3. Update paths:

```python
Config.DATA_ROOT = "/kaggle/input/your-dataset"
Config.CHECKPOINT_DIR = "/kaggle/working/checkpoints"
Config.LOG_DIR = "/kaggle/working/logs"
```

4. For resuming training, add previous checkpoint dataset as input

## Visualization Example

```python
# To display plots in Kaggle notebook
from IPython.display import Image, display

def show_plots():
    plot_files = [
        'loss_history.png',
        'accuracy_history.png',
        'learning_rate.png',
        'confusion_matrix.png',
        'roc_curve.png',
        'prediction_distribution.png',
        'metrics_summary.png'
    ]

    for plot in plot_files:
        print(f"\n{plot}:")
        display(Image(f'/kaggle/working/plots/{plot}'))

# Call after training
show_plots()
```
