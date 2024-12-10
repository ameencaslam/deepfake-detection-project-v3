import torch
from config.config import Config
from .metrics import MetricsCalculator
from .tracker import TrainingTracker
from utils.visualization import VisualizationUtils
import numpy as np
import os
import sys
import shutil

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device):
        self.device = device
        
        # Initialize model and move to device
        self.model = model.to(device)
        
        # Get optimizer and criterion
        self.optimizer = model.get_optimizer()
        self.criterion = model.get_criterion().to(device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Initialize tracker and metrics
        self.tracker = TrainingTracker(model.model_name)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = VisualizationUtils()
        
        # Create directories for plots
        os.makedirs('plots', exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.tracker.init_epoch(epoch, len(self.train_loader))
        
        running_loss = 0.0
        running_acc = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target.float())
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            acc = self.metrics_calculator.calculate_accuracy(output.squeeze(), target)
            
            # Update running metrics
            running_loss += loss.item()
            running_acc += acc
            
            # Update progress
            self.tracker.update_batch(
                batch_idx,
                loss.item(),
                acc,
                self.optimizer.param_groups[0]['lr'],
                running_loss,
                running_acc
            )
        
        # Update epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_acc / len(self.train_loader)
        self.tracker.update_epoch(epoch_loss, epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        all_outputs = []
        all_targets = []
        
        # Initialize validation progress bars
        self.tracker.init_validation(len(loader))
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.float())
                
                # Calculate batch accuracy
                acc = self.metrics_calculator.calculate_accuracy(output.squeeze(), target)
                
                # Update running metrics
                running_loss += loss.item()
                running_acc += acc
                
                # Update progress bars with batch metrics
                batch_loss = loss.item()
                batch_acc = acc
                self.tracker.update_validation_batch(
                    batch_idx,
                    batch_loss,
                    batch_acc,
                    running_loss,
                    running_acc
                )
                
                # Store predictions and targets for final metrics
                all_outputs.append(output.squeeze())
                all_targets.append(target)
        
        # Close validation progress bars
        self.tracker.close_validation()
        
        # Calculate final metrics
        val_loss = running_loss / len(loader)
        val_acc = running_acc / len(loader)
        metrics = self.metrics_calculator.calculate_epoch_metrics(
            all_outputs,
            all_targets
        )
        metrics['loss'] = val_loss
        metrics['accuracy'] = val_acc
        
        # Store validation metrics
        self.tracker.metrics['val']['loss'].append(val_loss)
        self.tracker.metrics['val']['accuracy'].append(metrics['accuracy'])
        self.tracker.metrics['val']['auc_roc'].append(metrics['auc_roc'])
        self.tracker.metrics['val']['f1_score'].append(metrics['f1_score'])
        self.tracker.metrics['val']['confusion_matrix'].append(metrics['confusion_matrix'])
        
        return metrics
    
    def train(self, resume=False):
        """Full training loop"""
        start_epoch = 0
        if resume:
            start_epoch = self.tracker.load_checkpoint(
                self.model,
                self.optimizer
            )
        
        for epoch in range(start_epoch, Config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
            
            # Save checkpoint - using validation loss as criterion
            is_best = val_metrics['loss'] < self.tracker.metrics['best_metrics']['val_loss']
            if is_best:
                self.tracker.metrics['best_metrics'].update({
                    'epoch': epoch,
                    'val_acc': val_metrics['accuracy'],
                    'val_loss': val_metrics['loss']
                })
                print(f"New best model! Val Loss: {val_metrics['loss']:.4f}")
            
            self.tracker.save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                is_best
            )
            
            # Plot training progress after each epoch
            self.visualizer.plot_training_history(self.tracker.metrics)
            self.visualizer.save_plots('plots')
    
    def test(self):
        """Test the model"""
        # Load best model
        self.tracker.load_checkpoint(
            self.model,
            self.optimizer,
            'best'
        )
        
        # Get predictions and metrics
        test_preds, test_probs, test_targets = self.get_predictions(self.test_loader)
        test_metrics = self.validate(self.test_loader)
        
        # Print results
        print("\nTest Results:")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test AUC-ROC: {test_metrics['auc_roc']:.4f}")
        print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
        
        # Plot test results
        self.visualizer.plot_confusion_matrix(test_targets, test_preds)
        self.visualizer.plot_roc_curve(test_targets, test_probs)
        self.visualizer.plot_prediction_distribution(test_probs)
        self.visualizer.plot_metrics_summary(test_metrics)
        self.visualizer.save_plots('plots')
        
        # Zip checkpoints
        if os.path.exists(Config.CHECKPOINT_DIR):
            shutil.make_archive('checkpoints', 'zip', Config.CHECKPOINT_DIR)
            print(f"\nCheckpoints saved to: checkpoints.zip")
            
            try:
                from IPython.display import display, FileLink
                if 'ipykernel' in sys.modules:  # Only if running in notebook
                    display(FileLink('checkpoints.zip'))
            except ImportError:
                pass  # Not in a notebook environment
        
        return test_metrics
    
    def get_predictions(self, loader):
        """Get model predictions"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output.squeeze())
                preds = (probs > 0.5).float()
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(target.numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_targets) 