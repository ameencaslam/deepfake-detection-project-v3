import os
import json
import torch
from tqdm import tqdm
from config.config import Config

class TrainingTracker:
    def __init__(self, model_name):
        self.model_name = model_name
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
        self.metrics_pbar = None
        self.val_pbar = None
        self.val_metrics_pbar = None
    
    def init_epoch(self, epoch, num_batches):
        """Initialize progress bars for new epoch"""
        # Clear any existing progress bars
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
        if self.metrics_pbar is not None:
            self.metrics_pbar.close()
            
        self.epoch_pbar = tqdm(
            total=num_batches,
            desc=f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}]',
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        self.metrics_pbar = tqdm(
            bar_format='{desc}',
            position=1,
            leave=True
        )
    
    def update_batch(self, batch_idx, loss, acc, lr, running_loss=None, running_acc=None):
        """Update batch progress with running averages"""
        self.epoch_pbar.update(1)
        
        # Calculate running averages if provided
        avg_loss = running_loss / (batch_idx + 1) if running_loss is not None else loss
        avg_acc = running_acc / (batch_idx + 1) if running_acc is not None else acc
        
        if batch_idx % Config.LOG_INTERVAL == 0:
            self.metrics_pbar.set_description_str(
                f'Loss: {loss:.4f} (avg: {avg_loss:.4f}) | Acc: {acc:.2f}% (avg: {avg_acc:.2f}%) | LR: {lr:.6f}'
            )
            
        # Store metrics
        self.metrics['train']['batch_loss'].append(loss)
        self.metrics['train']['batch_acc'].append(acc)
        self.metrics['train']['learning_rates'].append(lr)
    
    def update_epoch(self, epoch_loss, epoch_acc):
        """Update epoch metrics"""
        self.metrics['train']['epoch_loss'].append(epoch_loss)
        self.metrics['train']['epoch_acc'].append(epoch_acc)
        
        # Close progress bars
        self.epoch_pbar.close()
        self.metrics_pbar.close()
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_name': self.model_name,  # Save model architecture info
            'model_state_dict': model.module.state_dict() if Config.MULTI_GPU else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': self.metrics
        }
        
        # Save last checkpoint
        torch.save(state, os.path.join(checkpoint_dir, 'last.pth'))
        
        # Save best if needed
        if is_best:
            torch.save(state, os.path.join(checkpoint_dir, 'best.pth'))
            
        # Save metrics
        with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def load_checkpoint(self, model, optimizer, checkpoint_type='last'):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(
            Config.CHECKPOINT_DIR,
            self.model_name,
            f'{checkpoint_type}.pth'
        )
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            # Verify model architecture
            if 'model_name' not in checkpoint:
                print("Warning: Checkpoint doesn't contain model architecture info")
            elif checkpoint['model_name'] != self.model_name:
                raise ValueError(
                    f"Model architecture mismatch. Checkpoint is for {checkpoint['model_name']}, "
                    f"but trying to load into {self.model_name}"
                )
            
            try:
                if Config.MULTI_GPU:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.metrics = checkpoint['metrics']
                return checkpoint['epoch']
            except Exception as e:
                raise ValueError(f"Error loading checkpoint: {str(e)}")
        return 0 
    
    def init_validation(self, num_batches):
        """Initialize validation progress bars"""
        # Clear any existing validation progress bars
        if self.val_pbar is not None:
            self.val_pbar.close()
        if self.val_metrics_pbar is not None:
            self.val_metrics_pbar.close()
            
        self.val_pbar = tqdm(
            total=num_batches,
            desc='Validating',
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        self.val_metrics_pbar = tqdm(
            bar_format='{desc}',
            position=1,
            leave=True
        )
    
    def update_validation_batch(self, batch_idx, loss, acc, running_loss, running_acc):
        """Update validation progress with running averages"""
        self.val_pbar.update(1)
        
        # Calculate running averages
        avg_loss = running_loss / (batch_idx + 1)
        avg_acc = running_acc / (batch_idx + 1)
        
        self.val_metrics_pbar.set_description_str(
            f'Val Loss: {loss:.4f} (avg: {avg_loss:.4f}) | Val Acc: {acc:.2f}% (avg: {avg_acc:.2f}%)'
        )
    
    def close_validation(self):
        """Close validation progress bars"""
        if self.val_pbar is not None:
            self.val_pbar.close()
        if self.val_metrics_pbar is not None:
            self.val_metrics_pbar.close()