import os
import json
import torch
from tqdm import tqdm
from config.config import Config
import numpy as np
import mlflow
import mlflow.pytorch
from config.mlflow_config import MLflowConfig

class TrainingTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, self.model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize MLflow tracking
        self.experiment_name = MLflowConfig.get_experiment_name(model_name)
        mlflow.set_experiment(self.experiment_name)
        
        # Get or create run
        self.run = mlflow.active_run()
        if not self.run:
            self.run = mlflow.start_run()
        
        # Initialize training state
        self.training_state = {
            'completed_epochs': 0,
            'total_epochs': Config.NUM_EPOCHS,
            'best_val_loss': float('inf')
        }
        
        # Log initial state as metrics
        if mlflow.active_run():
            mlflow.log_metrics({
                'completed_epochs': self.training_state['completed_epochs'],
                'best_val_loss': self.training_state['best_val_loss']
            })
        
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
        if mlflow.active_run():
            # Try to load state from MLflow
            try:
                state = mlflow.get_run(mlflow.active_run().info.run_id).data.params.get('training_state')
                if state:
                    return json.loads(state)
            except:
                pass
        
        # Default state
        return {
            'completed_epochs': 0,
            'total_epochs': Config.NUM_EPOCHS,
            'best_val_loss': float('inf')
        }
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint and log to MLflow"""
        # Save local checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_name': self.model_name,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        # Save checkpoint locally
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'last.pth'))
        
        # Log to MLflow
        with mlflow.start_run(run_id=self.run.info.run_id):
            mlflow.pytorch.log_model(
                model,
                f"checkpoint_epoch_{epoch}",
                registered_model_name=f"{self.model_name}_v{epoch}" if is_best else None
            )
            
            # Log training state
            self.training_state['completed_epochs'] = epoch + 1
            mlflow.log_param('training_state', json.dumps(self.training_state))
            
            if is_best:
                mlflow.log_param('best_epoch', epoch)
                mlflow.log_metric('best_val_loss', self.training_state['best_val_loss'])
    
    def load_checkpoint(self, model, optimizer):
        """Load latest checkpoint from MLflow or local"""
        try:
            # Try loading from MLflow first
            with mlflow.start_run(run_id=self.run.info.run_id):
                latest_model = mlflow.pytorch.load_model(
                    f"runs:/{self.run.info.run_id}/checkpoint_epoch_{self.training_state['completed_epochs']-1}"
                )
                model.load_state_dict(latest_model.state_dict())
                
                print(f"Loaded checkpoint from MLflow. Completed epochs: {self.training_state['completed_epochs']}")
                return self.training_state['completed_epochs']
        except:
            # Fallback to local checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, 'last.pth')
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"Loaded local checkpoint. Completed epochs: {self.training_state['completed_epochs']}")
                return self.training_state['completed_epochs']
        
        print("No checkpoint found, starting fresh")
        return 0
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to MLflow"""
        with mlflow.start_run(run_id=self.run.info.run_id):
            mlflow.log_metrics(metrics, step=step)
    
    def log_batch_metrics(self, batch_idx, loss, acc, lr):
        """Log batch-level metrics"""
        with mlflow.start_run(run_id=self.run.info.run_id):
            mlflow.log_metrics({
                'batch_loss': loss,
                'batch_accuracy': acc,
                'learning_rate': lr
            }, step=batch_idx)
    
    def finish(self):
        """End MLflow run"""
        if mlflow.active_run():
            mlflow.end_run()
    
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
        """Update epoch metrics"""
        # Update training state
        self.training_state['completed_epochs'] += 1
        
        # Update best metrics if needed
        if epoch_loss < self.training_state['best_val_loss']:
            self.training_state['best_val_loss'] = epoch_loss
            self.training_state['best_epoch'] = self.training_state['completed_epochs']
        
        # Log metrics to MLflow
        if mlflow.active_run():
            mlflow.log_metrics({
                'epoch_loss': epoch_loss,
                'epoch_accuracy': epoch_acc,
                'best_val_loss': self.training_state['best_val_loss'],
                'best_epoch': float(self.training_state.get('best_epoch', 0)),
                'completed_epochs': float(self.training_state['completed_epochs'])
            }, step=self.training_state['completed_epochs'])
    
    def close_validation(self):
        """Close validation progress bars"""
        if self.val_pbar is not None:
            self.val_pbar.close()
    
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

    def _save_training_state(self):
        """Save training state as metrics instead of parameters"""
        if mlflow.active_run():
            mlflow.log_metrics({
                'completed_epochs': float(self.training_state['completed_epochs']),
                'best_val_loss': float(self.training_state['best_val_loss']),
                'best_epoch': float(self.training_state.get('best_epoch', 0))
            }, step=self.training_state['completed_epochs'])