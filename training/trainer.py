import torch
from config.config import Config
from .metrics import MetricsCalculator
from .tracker import TrainingTracker
from utils.visualization import VisualizationManager
import numpy as np
import os
import sys

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
        
        # Initialize visualization manager
        self.visualizer = None  # Will be initialized in train() or test()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.tracker.init_epoch(epoch, len(self.train_loader))
        
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        
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
            batch_size = data.size(0)
            num_samples += batch_size
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            
            # Calculate current averages
            current_avg_loss = running_loss / num_samples
            current_avg_acc = running_acc / num_samples
            
            # Update progress
            self.tracker.update_batch(
                batch_idx,
                loss.item(),
                acc,
                self.optimizer.param_groups[0]['lr'],
                current_avg_loss,
                current_avg_acc
            )
            
            # Update visualization learning rate
            if self.visualizer:
                self.visualizer.update_training_metrics(learning_rate=self.optimizer.param_groups[0]['lr'])
        
        # Calculate final epoch metrics
        epoch_loss = running_loss / num_samples
        epoch_acc = running_acc / num_samples
        self.tracker.update_epoch(epoch_loss, epoch_acc)
        
        # Update visualization metrics
        if self.visualizer:
            self.visualizer.update_training_metrics(
                epoch_loss=epoch_loss,
                epoch_acc=epoch_acc
            )
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        num_samples = 0
        all_outputs = []
        all_targets = []
        
        self.tracker.init_validation(len(loader))
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.float())
                
                acc = self.metrics_calculator.calculate_accuracy(output.squeeze(), target)
                
                batch_size = data.size(0)
                num_samples += batch_size
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                
                current_avg_loss = running_loss / num_samples
                current_avg_acc = running_acc / num_samples
                
                self.tracker.update_validation_batch(
                    batch_idx,
                    loss.item(),
                    acc,
                    current_avg_loss,
                    current_avg_acc
                )
                
                all_outputs.append(output.squeeze())
                all_targets.append(target)
        
        self.tracker.close_validation()
        
        # Calculate final metrics
        val_loss = running_loss / num_samples
        val_acc = running_acc / num_samples
        metrics = self.metrics_calculator.calculate_epoch_metrics(
            all_outputs,
            all_targets
        )
        metrics['loss'] = val_loss
        metrics['accuracy'] = val_acc
        
        # Update visualization metrics
        if self.visualizer:
            self.visualizer.update_validation_metrics(
                val_loss=val_loss,
                val_acc=val_acc,
                auc_roc=metrics['auc_roc'],
                f1_score=metrics['f1_score']
            )
        
        return metrics
    
    def get_predictions(self, loader):
        """Get model predictions for a data loader"""
        self.model.eval()
        all_probs = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_targets.extend(target.numpy())
        
        return np.array(all_preds), np.array(all_probs), np.array(all_targets)
    
    def train(self, resume=False):
        """Full training loop"""
        start_epoch = 0
        checkpoint_dir = None
        
        if resume:
            checkpoint_dir = self.tracker.get_checkpoint_dir()
            start_epoch = self.tracker.load_checkpoint(
                self.model,
                self.optimizer
            )
            if start_epoch >= Config.NUM_EPOCHS:
                print(f"\nTraining already completed ({start_epoch} epochs). Starting fresh.")
                self.tracker.reset_training()
                start_epoch = 0
                checkpoint_dir = None
        
        # Initialize visualization manager for training
        self.visualizer = VisualizationManager(
            model_name=self.model.model_name,
            mode='train',
            resume_from=checkpoint_dir
        )
        
        # Update visualizer with existing metrics if resuming
        if resume and checkpoint_dir:
            for epoch in range(start_epoch):
                if epoch < len(self.tracker.metrics['train']['epoch_loss']):
                    self.visualizer.update_training_metrics(
                        epoch_loss=self.tracker.metrics['train']['epoch_loss'][epoch],
                        epoch_acc=self.tracker.metrics['train']['epoch_acc'][epoch]
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
            
            # Save checkpoint
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
            
            # Update and save plots
            self.visualizer.plot_all()
        
        print(f"\nTraining plots saved to: {self.visualizer.model_plot_dir}")
    
    def test(self):
        """Test the model"""
        # Initialize visualization manager for testing
        self.visualizer = VisualizationManager(
            model_name=self.model.model_name,
            mode='test'
        )
        
        # Load best model
        self.tracker.load_checkpoint(
            self.model,
            self.optimizer,
            'best'
        )
        
        # Get predictions
        test_preds, test_probs, test_targets = self.get_predictions(self.test_loader)
        
        # Update visualization with test metrics
        self.visualizer.update_test_metrics(
            y_true=test_targets,
            y_pred=test_preds,
            y_prob=test_probs
        )
        
        # Generate and save test plots
        self.visualizer.plot_all()
        
        print(f"\nTest plots saved to: {self.visualizer.model_plot_dir}")
        
        # Return test metrics
        return {
            'accuracy': (test_preds == test_targets).mean(),
            'predictions': test_preds,
            'probabilities': test_probs,
            'targets': test_targets
        } 