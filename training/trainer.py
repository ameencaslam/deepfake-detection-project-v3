import torch
from config.config import Config
from .metrics import MetricsCalculator
from .tracker import TrainingTracker
from utils.visualization import VisualizationUtils
import numpy as np

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, rank=0):
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        
        # Get optimizer and criterion before DDP wrapping
        self.optimizer = model.get_optimizer()
        self.criterion = model.get_criterion().to(self.device)
        
        # Store model name before DDP wrapping
        self.model_name = model.model_name
        
        # Wrap model in DDP
        self.model = model.to(self.device)
        if torch.cuda.device_count() > 1:
            self.model = DDP(model, device_ids=[rank])
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Initialize tracker and metrics
        self.tracker = TrainingTracker(self.model_name)
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = VisualizationUtils()
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)
        
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
            
            # Update progress (only on rank 0)
            if self.rank == 0:
                self.tracker.update_batch(
                    batch_idx,
                    loss.item(),
                    acc,
                    self.optimizer.param_groups[0]['lr'],
                    running_loss,
                    running_acc
                )
        
        # Update epoch metrics (only on rank 0)
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = running_acc / len(self.train_loader)
        if self.rank == 0:
            self.tracker.update_epoch(epoch_loss, epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        all_outputs = []
        all_targets = []
        
        # Initialize validation progress bars (only on rank 0)
        if self.rank == 0:
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
                
                # Update progress bars with batch metrics (only on rank 0)
                if self.rank == 0:
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
        
        # Close validation progress bars (only on rank 0)
        if self.rank == 0:
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
        
        # Store validation metrics (only on rank 0)
        if self.rank == 0:
            self.tracker.metrics['val']['loss'].append(val_loss)
            self.tracker.metrics['val']['accuracy'].append(metrics['accuracy'])
            self.tracker.metrics['val']['auc_roc'].append(metrics['auc_roc'])
            self.tracker.metrics['val']['f1_score'].append(metrics['f1_score'])
            self.tracker.metrics['val']['confusion_matrix'].append(metrics['confusion_matrix'])
        
        return metrics
    
    def train(self, resume=False):
        """Full training loop"""
        start_epoch = 0
        if resume and self.rank == 0:
            start_epoch = self.tracker.load_checkpoint(
                self.model,
                self.optimizer
            )
        
        for epoch in range(start_epoch, Config.NUM_EPOCHS):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(self.val_loader)
            
            # Print metrics and save checkpoints only on rank 0
            if self.rank == 0:
                print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"Val AUC-ROC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1_score']:.4f}")
                
                # Save checkpoint - now using validation loss as criterion
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