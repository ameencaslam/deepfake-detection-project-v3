import os
import json
from datetime import datetime
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from IPython.display import display, Image
from config.config import Config

class VisualizationManager:
    def __init__(self, model_name, mode='train', resume_from=None):
        """
        Initialize visualization manager
        Args:
            model_name: Name of the model being trained/tested
            mode: 'train' or 'test'
            resume_from: Path to previous training checkpoint (if resuming)
        """
        self.model_name = model_name
        self.mode = mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model-specific plot directory
        self.base_plot_dir = os.path.join(Config.LOG_DIR, 'plots')
        self.model_plot_dir = os.path.join(self.base_plot_dir, f'{model_name}_{self.timestamp}')
        os.makedirs(self.model_plot_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'train': {
                'epoch_loss': [],
                'epoch_acc': [],
                'learning_rates': []
            },
            'val': {
                'loss': [],
                'accuracy': [],
                'auc_roc': [],
                'f1_score': []
            },
            'test': {
                'confusion_matrix': None,
                'roc_data': None,
                'predictions': None
            }
        }
        
        # Load previous metrics if resuming
        if resume_from and os.path.exists(resume_from):
            self._load_previous_metrics(resume_from)
            
        # Set style
        plt.style.use('seaborn')
        self.colors = {
            'train': '#2ecc71',
            'val': '#e74c3c',
            'test': '#3498db'
        }
    
    def _load_previous_metrics(self, checkpoint_dir):
        """Load metrics from previous training"""
        metrics_file = os.path.join(checkpoint_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    prev_metrics = json.load(f)
                # Merge previous metrics with current
                for key in ['train', 'val']:
                    if key in prev_metrics:
                        for metric in prev_metrics[key]:
                            if metric in self.metrics[key]:
                                self.metrics[key][metric].extend(prev_metrics[key][metric])
                print(f"Loaded previous metrics from {metrics_file}")
            except Exception as e:
                print(f"Error loading previous metrics: {str(e)}")
    
    def save_metrics(self):
        """Save current metrics to JSON"""
        metrics_file = os.path.join(self.model_plot_dir, 'metrics.json')
        try:
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f)
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
    
    def update_training_metrics(self, epoch_loss=None, epoch_acc=None, learning_rate=None):
        """Update training metrics"""
        if epoch_loss is not None:
            self.metrics['train']['epoch_loss'].append(epoch_loss)
        if epoch_acc is not None:
            self.metrics['train']['epoch_acc'].append(epoch_acc)
        if learning_rate is not None:
            self.metrics['train']['learning_rates'].append(learning_rate)
    
    def update_validation_metrics(self, val_loss=None, val_acc=None, auc_roc=None, f1_score=None):
        """Update validation metrics"""
        if val_loss is not None:
            self.metrics['val']['loss'].append(val_loss)
        if val_acc is not None:
            self.metrics['val']['accuracy'].append(val_acc)
        if auc_roc is not None:
            self.metrics['val']['auc_roc'].append(auc_roc)
        if f1_score is not None:
            self.metrics['val']['f1_score'].append(f1_score)
    
    def update_test_metrics(self, y_true, y_pred, y_prob):
        """Update test metrics"""
        self.metrics['test']['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        self.metrics['test']['roc_data'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 
                                          'auc': auc(fpr, tpr)}
        self.metrics['test']['predictions'] = {'true': y_true.tolist(), 
                                             'pred': y_pred.tolist(), 
                                             'prob': y_prob.tolist()}
    
    def plot_training_progress(self):
        """Plot training progress metrics separately"""
        # Plot loss
        if self.metrics['train']['epoch_loss']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['train']['epoch_loss'], 
                    color=self.colors['train'], label='Train Loss')
            if self.metrics['val']['loss']:
                plt.plot(self.metrics['val']['loss'], 
                        color=self.colors['val'], label='Val Loss')
            plt.title(f'Loss History - {self.model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(self.model_plot_dir, 'loss_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot accuracy
        if self.metrics['train']['epoch_acc']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['train']['epoch_acc'], 
                    color=self.colors['train'], label='Train Acc')
            if self.metrics['val']['accuracy']:
                plt.plot(self.metrics['val']['accuracy'], 
                        color=self.colors['val'], label='Val Acc')
            plt.title(f'Accuracy History - {self.model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(self.model_plot_dir, 'accuracy_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot learning rate
        if self.metrics['train']['learning_rates']:
            plt.figure(figsize=(10, 6))
            plt.plot(self.metrics['train']['learning_rates'], 
                    color=self.colors['train'])
            plt.title(f'Learning Rate - {self.model_name}')
            plt.xlabel('Iteration')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            save_path = os.path.join(self.model_plot_dir, 'learning_rate.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot validation metrics
        if self.metrics['val']['auc_roc'] and self.metrics['val']['f1_score']:
            plt.figure(figsize=(10, 6))
            epochs = range(len(self.metrics['val']['auc_roc']))
            plt.plot(epochs, self.metrics['val']['auc_roc'], 
                    color='purple', label='AUC-ROC')
            plt.plot(epochs, self.metrics['val']['f1_score'], 
                    color='orange', label='F1-Score')
            plt.title(f'Validation Metrics - {self.model_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(self.model_plot_dir, 'validation_metrics.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_test_results(self):
        """Plot test results separately"""
        if not self.metrics['test']['confusion_matrix'] is not None:
            print("No test metrics available")
            return
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.metrics['test']['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        save_path = os.path.join(self.model_plot_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        roc_data = self.metrics['test']['roc_data']
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                color=self.colors['test'], 
                label=f'ROC curve (AUC = {roc_data["auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {self.model_name}')
        plt.legend(loc="lower right")
        save_path = os.path.join(self.model_plot_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction Distribution
        plt.figure(figsize=(8, 6))
        predictions = self.metrics['test']['predictions']
        sns.histplot(predictions['prob'], bins=50)
        plt.title(f'Prediction Distribution - {self.model_name}')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        save_path = os.path.join(self.model_plot_dir, 'prediction_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def display_plots(self):
        """Display all plots in the current model directory"""
        print(f"\nDisplaying plots from: {self.model_plot_dir}")
        for plot_path in sorted(glob.glob(os.path.join(self.model_plot_dir, '*.png'))):
            print(f"Displaying: {os.path.basename(plot_path)}")
            display(Image(filename=plot_path))
    
    def plot_all(self):
        """Plot all available visualizations based on mode"""
        if self.mode == 'train':
            self.plot_training_progress()
        elif self.mode == 'test':
            self.plot_test_results()
        self.save_metrics()
        
        # Display plots if in Kaggle environment
        if os.path.exists('/kaggle'):
            self.display_plots() 