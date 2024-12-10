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
        self.val_pbar = None
    
    def init_epoch(self, epoch, num_batches):
        """Initialize progress bars for new epoch"""
        # Clear any existing progress bars
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
        
        desc = f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}]'
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
        avg_loss = f"{running_loss / (batch_idx + 1):.4f}" if running_loss is not None else "?"
        avg_acc = f"{running_acc / (batch_idx + 1):.2f}" if running_acc is not None else "?"
        
        # Update metrics in progress bar
        self.epoch_pbar.set_postfix({
            'loss': f"{loss:.4f}",
            'avg_loss': avg_loss,
            'acc': f"{acc:.2f}%",
            'avg_acc': f"{avg_acc}%",
            'lr': f"{lr:.6f}"
        }, refresh=True)
        
        # Store metrics
        self.metrics['train']['batch_loss'].append(loss)
        self.metrics['train']['batch_acc'].append(acc)
        self.metrics['train']['learning_rates'].append(lr)
    
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
        """Update epoch metrics"""
        self.metrics['train']['epoch_loss'].append(epoch_loss)
        self.metrics['train']['epoch_acc'].append(epoch_acc)
        
        # Close progress bars
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
    
    def close_validation(self):
        """Close validation progress bars"""
        if self.val_pbar is not None:
            self.val_pbar.close()
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        state = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': model.state_dict(),
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
                self.metrics = checkpoint['metrics']
                total_epochs = checkpoint['epoch'] + 1
                print(f"Successfully loaded checkpoint. Total epochs completed: {total_epochs}")
                return checkpoint['epoch']
            except Exception as e:
                raise ValueError(f"Error loading checkpoint: {str(e)}")
        print("No checkpoint found, starting from scratch")
        return 0

    def save_to_zip(self):
        """Save checkpoints to zip file"""
        checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)
        if os.path.exists(checkpoint_dir):
            import shutil
            zip_path = os.path.join(Config.CHECKPOINT_DIR, 'checkpoints.zip')
            shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', Config.CHECKPOINT_DIR)
            print(f"\nCheckpoints saved to: {zip_path}")