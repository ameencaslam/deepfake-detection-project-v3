import torch
from config.config import Config
from .metrics import MetricsCalculator
from .tracker import TrainingTracker
from utils.visualization import VisualizationUtils
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.optimizer = model.get_optimizer()
        self.criterion = model.get_criterion()
        self.tracker = TrainingTracker(model.model_name)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = VisualizationUtils()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(Config.GPU_IDS[0])
        
        # Move model to device
        self.model = model.prepare_model()
        self.criterion = self.criterion.to(self.device)
    
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
            
            # Update metrics
            running_loss += loss.item()
            running_acc += acc
            
            # Update progress
            self.tracker.update_batch(
                batch_idx,
                loss.item(),
                acc,
                self.optimizer.param_groups[0]['lr']
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
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target.float())
                
                # Store predictions and targets
                all_outputs.append(output.squeeze())
                all_targets.append(target)
                running_loss += loss.item()
        
        # Calculate metrics
        val_loss = running_loss / len(loader)
        metrics = self.metrics_calculator.calculate_epoch_metrics(
            all_outputs,
            all_targets
        )
        metrics['loss'] = val_loss
        
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
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_metrics = self.validate(self.val_loader)
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.metrics_calculator.print_metrics(val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.tracker.metrics['best_metrics']['val_acc']
            if is_best:
                self.tracker.metrics['best_metrics'].update({
                    'epoch': epoch,
                    'val_acc': val_metrics['accuracy'],
                    'val_loss': val_metrics['loss']
                })
            
            self.tracker.save_checkpoint(
                self.model,
                self.optimizer,
                epoch,
                is_best
            )
            
            # Plot training progress
            self.visualizer.plot_training_history(self.tracker.metrics)
        
        # After training completes
        # Plot final training curves
        self.visualizer.plot_training_history(self.tracker.metrics)
        
        # Get and plot validation predictions
        val_preds, val_probs, val_targets = self.get_predictions(self.val_loader)
        self.visualizer.plot_confusion_matrix(val_targets, val_preds)
        self.visualizer.plot_roc_curve(val_targets, val_probs)
        self.visualizer.plot_prediction_distribution(val_probs)
        
        # Zip checkpoints
        import os
        import shutil
        if os.path.exists(Config.CHECKPOINT_DIR):
            shutil.make_archive('checkpoints', 'zip', Config.CHECKPOINT_DIR)
            print(f"\nCheckpoints saved to: checkpoints.zip")
            
            try:
                from IPython.display import FileLink
                display(FileLink('checkpoints.zip'))
            except ImportError:
                pass
    
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
        self.metrics_calculator.print_metrics(test_metrics)
        
        # Plot test results
        self.visualizer.plot_confusion_matrix(test_targets, test_preds)
        self.visualizer.plot_roc_curve(test_targets, test_probs)
        self.visualizer.plot_prediction_distribution(test_probs)
        self.visualizer.plot_metrics_summary(test_metrics)
        
        # Zip checkpoints
        import os
        import shutil
        if os.path.exists(Config.CHECKPOINT_DIR):
            shutil.make_archive('checkpoints', 'zip', Config.CHECKPOINT_DIR)
            print(f"\nCheckpoints saved to: checkpoints.zip")
            
            try:
                from IPython.display import FileLink
                display(FileLink('checkpoints.zip'))
            except ImportError:
                pass
        
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