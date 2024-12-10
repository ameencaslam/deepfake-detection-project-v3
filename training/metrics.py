import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

class MetricsCalculator:
    @staticmethod
    def calculate_accuracy(outputs, targets):
        """Calculate accuracy for batch"""
        with torch.no_grad():
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct = (predictions == targets).float().sum()
            accuracy = correct / targets.size(0) * 100
            return accuracy.item()
    
    @staticmethod
    def calculate_epoch_metrics(all_outputs, all_targets):
        """Calculate metrics for entire epoch"""
        with torch.no_grad():
            # Convert to numpy
            outputs = torch.cat(all_outputs).cpu().numpy()
            targets = torch.cat(all_targets).cpu().numpy()
            
            # Calculate predictions
            predictions = (1 / (1 + np.exp(-outputs)) > 0.5).astype(np.float32)
            
            # Calculate metrics
            accuracy = (predictions == targets).mean() * 100
            auc_roc = roc_auc_score(targets, 1 / (1 + np.exp(-outputs)))
            f1 = f1_score(targets, predictions)
            conf_matrix = confusion_matrix(targets, predictions)
            
            return {
                'accuracy': accuracy,
                'auc_roc': auc_roc,
                'f1_score': f1,
                'confusion_matrix': conf_matrix.tolist()
            }
    
    @staticmethod
    def print_metrics(metrics):
        """Print metrics in a formatted way"""
        print("\nMetrics:")
        print(f"Accuracy: {metrics['accuracy']:.2f}%")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(np.array(metrics['confusion_matrix'])) 