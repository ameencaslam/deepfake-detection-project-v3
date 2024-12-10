import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config.config import Config

class VisualizationUtils:
    @staticmethod
    def plot_training_history(metrics, save_dir="/kaggle/working/plots"):
        """Plot training and validation metrics history"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['train']['epoch_loss'], label='Training Loss')
        plt.plot(metrics['val']['loss'], label='Validation Loss')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'loss_history.png'))
        plt.close()
        
        # Accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['train']['epoch_acc'], label='Training Accuracy')
        plt.plot(metrics['val']['accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'accuracy_history.png'))
        plt.close()
        
        # Learning rate plot
        plt.figure(figsize=(10, 5))
        plt.plot(metrics['train']['learning_rates'])
        plt.title('Learning Rate Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
        plt.close()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_dir="/kaggle/working/plots"):
        """Plot confusion matrix"""
        os.makedirs(save_dir, exist_ok=True)
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
        plt.close()
    
    @staticmethod
    def plot_roc_curve(y_true, y_prob, save_dir="/kaggle/working/plots"):
        """Plot ROC curve"""
        os.makedirs(save_dir, exist_ok=True)
        
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
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()
    
    @staticmethod
    def plot_prediction_distribution(y_prob, save_dir="/kaggle/working/plots"):
        """Plot distribution of prediction probabilities"""
        os.makedirs(save_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        sns.histplot(data=y_prob, bins=50)
        plt.title('Distribution of Prediction Probabilities')
        plt.xlabel('Probability of being Fake')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'prediction_distribution.png'))
        plt.close()
    
    @staticmethod
    def plot_metrics_summary(metrics, save_dir="/kaggle/working/plots"):
        """Plot summary of key metrics"""
        os.makedirs(save_dir, exist_ok=True)
        
        metrics_list = ['accuracy', 'auc_roc', 'f1_score']
        values = [metrics[m] for m in metrics_list]
        
        plt.figure(figsize=(10, 5))
        bars = plt.bar(metrics_list, values)
        plt.title('Model Performance Metrics')
        plt.ylim(0, 100 if max(values) > 1 else 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
        
        plt.grid(True, axis='y')
        plt.savefig(os.path.join(save_dir, 'metrics_summary.png'))
        plt.close() 