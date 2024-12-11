import os
import json
import torch
from tqdm import tqdm
from config.config import Config
import numpy as np

class TrainingTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize or load training state
        self.training_state_file = os.path.join(self.checkpoint_dir, 'training_state.json')
        self.training_state = self._load_training_state()
        
        self.metrics = {
            'train': {
                'batch_loss': [],
                'epoch_loss': [],
                'batch_acc': [],
                'epoch_acc': [],
                'learning_rates': []
            },
            'val': {
                'loss': [],
                'accuracy': [],
                'auc_roc': [],
                'f1_score': [],
                'confusion_matrix': []
            },
            'best_metrics': {
                'epoch': 0,
                'val_acc': 0,
                'val_loss': float('inf')
            }
        }
        
        # Create progress bars
        self.epoch_pbar = None
        self.val_pbar = None
    
    def _load_training_state(self):
        """Load or initialize training state"""
        if os.path.exists(self.training_state_file):
            with open(self.training_state_file, 'r') as f:
                state = json.load(f)
                # Update total_epochs if it changed but keep completed_epochs
                state['total_epochs'] = Config.NUM_EPOCHS
                return state
        return {
            'completed_epochs': 0,
            'total_epochs': Config.NUM_EPOCHS,
            'best_val_loss': float('inf')
        }

    def _save_training_state(self):
        """Save training state"""
        with open(self.training_state_file, 'w') as f:
            json.dump(self.training_state, f, indent=4)
    
    def init_epoch(self, epoch, num_batches):
        """Initialize progress bars for new epoch"""
        # Clear any existing progress bars
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
        
        # Adjust epoch display to be 1-based
        current_epoch = epoch + 1
        desc = f'Epoch [{current_epoch}/{Config.NUM_EPOCHS}]'
        bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        
        self.epoch_pbar = tqdm(
            total=num_batches,
            desc=desc,
            bar_format=bar_format,
            leave=True
        )
        # Initialize postfix dict for metrics
        self.epoch_pbar.set_postfix({
            'loss': '?',
            'avg_loss': '?',
            'acc': '?',
            'avg_acc': '?',
            'lr': '?'
        })
    
    def update_batch(self, batch_idx, loss, acc, lr, running_loss=None, running_acc=None):
        """Update batch progress with running averages"""
        self.epoch_pbar.update(1)
        
        # Calculate running averages if provided
        avg_loss = f"{running_loss:.4f}" if running_loss is not None else "?"
        avg_acc = f"{running_acc:.2f}" if running_acc is not None else "?"
        
        # Update metrics in progress bar
        self.epoch_pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'avg_loss': avg_loss,
            'acc': f"{acc:.2f}%",
            'avg_acc': f"{avg_acc}%",
            'lr': f"{lr:.6f}"
        }, refresh=True)
        
        # Store metrics
        self.metrics['train']['batch_loss'].append(float(loss))
        self.metrics['train']['batch_acc'].append(float(acc))
        self.metrics['train']['learning_rates'].append(float(lr))
    
    def init_validation(self, num_batches):
        """Initialize validation progress bars"""
        # Clear any existing progress bars
        if self.val_pbar is not None:
            self.val_pbar.close()
        
        bar_format = 'Validating: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        
        self.val_pbar = tqdm(
            total=num_batches,
            bar_format=bar_format,
            leave=True
        )
        # Initialize postfix dict for metrics
        self.val_pbar.set_postfix({
            'loss': '?',
            'avg_loss': '?',
            'acc': '?',
            'avg_acc': '?'
        })
    
    def update_validation_batch(self, batch_idx, loss, acc, running_loss, running_acc):
        """Update validation progress with running averages"""
        self.val_pbar.update(1)
        
        # Calculate running averages
        avg_loss = running_loss / (batch_idx + 1)
        avg_acc = running_acc / (batch_idx + 1)
        
        # Update metrics in progress bar
        self.val_pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'avg_loss': f"{avg_loss:.4f}",
            'acc': f"{acc:.2f}%",
            'avg_acc': f"{avg_acc:.2f}%"
        }, refresh=True)
    
    def update_epoch(self, epoch_loss, epoch_acc):
        """Update epoch metrics and training state"""
        # Convert to float to ensure JSON serialization
        self.metrics['train']['epoch_loss'].append(float(epoch_loss))
        self.metrics['train']['epoch_acc'].append(float(epoch_acc))
        
        # Update training state
        self.training_state['completed_epochs'] += 1
        self._save_training_state()
        
        # Close progress bars
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None
    
    def close_validation(self):
        """Close validation progress bars"""
        if self.val_pbar is not None:
            self.val_pbar.close()
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Convert all numpy arrays and tensors to regular Python types for JSON serialization
        metrics_json = {}
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                metrics_json[key] = {}
                for k, v in value.items():
                    if isinstance(v, (list, np.ndarray)):
                        metrics_json[key][k] = [float(x) if isinstance(x, (np.number, torch.Tensor)) else x for x in v]
                    else:
                        metrics_json[key][k] = float(v) if isinstance(v, (np.number, torch.Tensor)) else v
            else:
                metrics_json[key] = float(value) if isinstance(value, (np.number, torch.Tensor)) else value
        
        state = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics_json
        }
        
        # Save last checkpoint
        torch.save(state, os.path.join(checkpoint_dir, 'last.pth'))
        
        # Save best if needed
        if is_best:
            torch.save(state, os.path.join(checkpoint_dir, 'best.pth'))
        
        # Save metrics separately to ensure consistency
        metrics_file = os.path.join(checkpoint_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=4)
    
    def load_checkpoint(self, model, optimizer, checkpoint_type='last'):
        """Load model checkpoint and return next epoch number"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{checkpoint_type}.pth')
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            
            # Verify model architecture
            if 'model_name' in checkpoint and checkpoint['model_name'] != self.model_name:
                raise ValueError(
                    f"Model architecture mismatch. Checkpoint is for {checkpoint['model_name']}, "
                    f"but trying to load into {self.model_name}"
                )
            
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load metrics from metrics.json instead of checkpoint
                metrics_file = os.path.join(self.checkpoint_dir, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        self.metrics = json.load(f)
                else:
                    self.metrics = checkpoint['metrics']  # Fallback to checkpoint metrics
                
                # Load training state
                if os.path.exists(self.training_state_file):
                    self.training_state = self._load_training_state()  # This will update total_epochs
                
                completed_epochs = self.training_state['completed_epochs']
                remaining_epochs = Config.NUM_EPOCHS - completed_epochs
                print(f"Successfully loaded checkpoint. Completed epochs: {completed_epochs}")
                print(f"Remaining epochs: {remaining_epochs}")
                return completed_epochs
            except Exception as e:
                raise ValueError(f"Error loading checkpoint: {str(e)}")
        
        print("No checkpoint found, starting fresh training...")
        return 0

    def reset_training(self):
        """Reset training state"""
        self.training_state = {
            'completed_epochs': 0,
            'total_epochs': Config.NUM_EPOCHS,
            'best_val_loss': float('inf')
        }
        self._save_training_state()
        
        # Also reset metrics
        self.metrics = {
            'train': {
                'batch_loss': [],
                'epoch_loss': [],
                'batch_acc': [],
                'epoch_acc': [],
                'learning_rates': []
            },
            'val': {
                'loss': [],
                'accuracy': [],
                'auc_roc': [],
                'f1_score': [],
                'confusion_matrix': []
            },
            'best_metrics': {
                'epoch': 0,
                'val_acc': 0,
                'val_loss': float('inf')
            }
        }

    def save_to_zip(self):
        """This method is deprecated. Use utils.checkpoint_utils.zip_checkpoints instead."""
        from utils.checkpoint_utils import zip_checkpoints
        print("\nWarning: This method is deprecated. Use utils.checkpoint_utils.zip_checkpoints instead.")
        return zip_checkpoints()

    def get_checkpoint_dir(self):
        """Get the path to the checkpoint directory"""
        checkpoint_dir = os.path.join("/kaggle/working/checkpoints", self.model_name)
        return checkpoint_dir if os.path.exists(checkpoint_dir) else None