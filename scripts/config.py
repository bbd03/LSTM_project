from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"
LOGS_DIR = ROOT_DIR / "logs"
LOG_FILE = LOGS_DIR / "training.log"


GOOGLE_NEWS_MODEL = MODELS_DIR / "GoogleNews-vectors-negative300.bin"
TRAIN_DATA = DATA_DIR / "raw" / "Genre Classification Dataset" / "train_data.txt"
TEST_DATA = DATA_DIR / "raw" / "Genre Classification Dataset" / "test_data.txt"
PROCESSED_DATA = DATA_DIR / "processed"

class Config:
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 17
    EMBEDDING_DIM = 300
    HIDDEN_SIZE = 256
    NUM_CLASSES = 27
    MAX_SEQ_LENGTH = None
    LOG_INTERVAL = 100
    TEST_NUM = 0.1
    USE_CLASS_WEIGHTS = True
    MAX_CLASS_WEIGHT = 1.5
    MIN_CLASS_WEIGHT = 1
    USE_CLIP_GRAD = True
    GRAD_CLIP_NORM = 8.0
    USE_SAMPLER = False
    AUGMENT_THRESHOLD = 800
    SWAPS_PER_SAMPLE   = 8
    COPIES_PER_SAMPLE  = 10
