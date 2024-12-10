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
        self.batch_pbar = None
        self.val_pbar = None
    
    def init_epoch(self, epoch, num_batches):
        """Initialize progress bars for new epoch"""
        self.epoch_pbar = tqdm(
            total=num_batches,
            desc=f'Epoch [{epoch+1}/{Config.NUM_EPOCHS}]',
            position=0
        )
        
        self.metrics_pbar = tqdm(
            bar_format='{desc}',
            position=1
        )
    
    def update_batch(self, batch_idx, loss, acc, lr):
        """Update batch progress"""
        self.epoch_pbar.update(1)
        
        if batch_idx % Config.LOG_INTERVAL == 0:
            self.metrics_pbar.set_description_str(
                f'Loss: {loss:.4f} | Acc: {acc:.2f}% | LR: {lr:.6f}'
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