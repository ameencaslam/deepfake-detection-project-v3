class Config:
    # Dataset
    DATA_ROOT = "path/to/dataset"  # Update this with your dataset path
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Training
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    NUM_WORKERS = 4
    
    # Model
    IMAGE_SIZE = {
        'efficientnet': 224,
        'swin': 256,
        'two_stream': 224,
        'xception': 299,
        'cnn_transformer': 256,
        'cross_attention': 256
    }
    
    # Optimization
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Logging
    LOG_INTERVAL = 10
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # Model Specific
    MODEL_CONFIGS = {
        'efficientnet': {'version': 'b3'},
        'swin': {'embed_dim': 96, 'depths': [2, 2, 6, 2]},
        'two_stream': {'backbone': 'resnet18'},
        'xception': {},
        'cnn_transformer': {'num_heads': 8},
        'cross_attention': {'num_heads': 8}
    } 