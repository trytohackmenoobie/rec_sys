import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DB_PATH = os.path.join(BASE_DIR, 'notifications.db')
    
    # Dataset/training parameters - OPTIMIZED for better performance
    SEQ_LENGTH = 10   # Reduced sequence length
    BATCH_SIZE = 32   # Increased batch size
    EPOCHS = 30       # Back to working configuration
    LEARNING_RATE = 0.001  # Back to working learning rate
    
    # Dataset splits
    TRAIN_SPLIT = 'train'
    VAL_SPLIT = 'validation'
    TEST_SPLIT = 'test'
    
    # Model architecture dimensions - WORKING CONFIGURATION
    USE_IMPROVED_MODEL = False  # Back to simple working model
    POI_EMBEDDING_DIM = 32      # Back to working embeddings
    USER_EMBEDDING_DIM = 16     # Back to working user embeddings
    HIDDEN_DIM = 64             # Back to working hidden dimension
    NUM_ATTENTION_HEADS = 4     # Keep for future use
    NUM_GRU_LAYERS = 2          # Keep for future use
    
    # User-related enumerations
    PERSONALITY_TRAITS = ['extrovert', 'conscientious', 'neurotic', 'agreeable', 'open']
    FOOD_PREFERENCES = ['sushi', 'american', 'cafe', 'russian', 'italian', 'chinese', 'georgian']
    
    # FourSquare dataset specific settings
    DATASET_NAME = 'w11wo/FourSquare-Moscow-POI'
    
    # Training optimization - SIMPLIFIED
    DROPOUT_RATE = 0.1  # Reduced dropout
    WEIGHT_DECAY = 1e-4  # L2 regularization
    GRADIENT_CLIPPING = 1.0  # Gradient clipping threshold
    LOG_INTERVAL = 100  # Log every N batches
    
    # Loss function settings - SIMPLIFIED
    LOSS_TYPE = 'cross_entropy'  # Simple cross entropy loss only
    
    # Evaluation metrics
    EVAL_K_VALUES = [1, 5, 10, 20]  # K values for evaluation metrics

config = Config()