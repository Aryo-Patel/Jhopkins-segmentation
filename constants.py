import torch

ON_ARM = True

if ON_ARM:
  DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
else:
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_PIXEL_VALUE = 65535
BUCKET_NAME = "cindy-profiling"
RAW_BASE_PATH = "TargetActivationAdvanced/"
CLEAN_BASE_PATH = "Data/"
RUNS_BASE_PATH = "Runs/"

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
NUM_WORKERS = 4
BATCH_SIZE = 16
FEATURES = [32, 64] #[64, 128, 256, 512]
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500

SAVE_STATS = True
