import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config.config import Config

class VisualizationUtils:
    def __init__(self):
        plt.style.use('seaborn')
    
    def save_plots(self, save_dir):
        """Save all current plots to directory"""
        plt.close('all')  # Close any existing plots
        
        # Save current figure
        if plt.get_fignums():
            for i, fig in enumerate(plt.get_fignums()):
                plt.figure(fig)
                plt.savefig(os.path.join(save_dir, f'plot_{i}.png'))
                plt.close(fig)
    
    def plot_training_history(self, metrics):
        """Plot training history"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot loss
        axes[0, 0].plot(metrics['train']['epoch_loss'], label='Train Loss')
        if metrics['val']['loss']:
            axes[0, 0].plot(metrics['val']['loss'], label='Val Loss')
        axes[0, 0].set_title('Loss History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Plot accuracy
        axes[0, 1].plot(metrics['train']['epoch_acc'], label='Train Acc')
        if metrics['val']['accuracy']:
            axes[0, 1].plot(metrics['val']['accuracy'], label='Val Acc')
        axes[0, 1].set_title('Accuracy History')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        
        # Plot learning rate
        if metrics['train']['learning_rates']:
            axes[1, 0].plot(metrics['train']['learning_rates'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Batch')
            axes[1, 0].set_ylabel('Learning Rate')
        
        # Plot validation metrics
        if metrics['val']['auc_roc']:
            epochs = range(len(metrics['val']['auc_roc']))
            axes[1, 1].plot(epochs, metrics['val']['auc_roc'], label='AUC-ROC')
            axes[1, 1].plot(epochs, metrics['val']['f1_score'], label='F1-Score')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_prob):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        return plt.gcf()
    
    def plot_prediction_distribution(self, probabilities):
        """Plot prediction probability distribution"""
        plt.figure(figsize=(8, 6))
        sns.histplot(probabilities, bins=50)
        plt.title('Prediction Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Count')
        return plt.gcf()
    
    def plot_metrics_summary(self, metrics):
        """Plot summary of metrics"""
        plt.figure(figsize=(10, 6))
        metrics_to_plot = {
            'Accuracy': metrics['accuracy'],
            'AUC-ROC': metrics['auc_roc'],
            'F1-Score': metrics['f1_score']
        }
        
        plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
        plt.title('Metrics Summary')
        plt.ylim([0, 100])
        
        # Add value labels on top of bars
        for i, (metric, value) in enumerate(metrics_to_plot.items()):
            plt.text(i, value + 1, f'{value:.2f}', ha='center')
        
        return plt.gcf() 