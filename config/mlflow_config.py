import os
import mlflow

class MLflowConfig:
    # MLflow tracking URI (local or remote)
    TRACKING_URI = "mlflow"  # For local sqlite database
    
    # Experiment naming
    EXPERIMENT_NAME_PREFIX = "deepfake_detection"
    
    # Artifact paths
    ARTIFACT_PATH = "models"
    
    # Tags for organization
    COMMON_TAGS = {
        "project": "deepfake_detection",
        "framework": "pytorch"
    }
    
    @staticmethod
    def setup_mlflow():
        """Setup MLflow tracking"""
        mlflow.set_tracking_uri(MLflowConfig.TRACKING_URI)
        
    @staticmethod
    def get_experiment_name(model_name):
        """Get experiment name for model"""
        return f"{MLflowConfig.EXPERIMENT_NAME_PREFIX}_{model_name}" 