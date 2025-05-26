import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 4
MAX_IMAGES = 5
EPOCHS = 5

DATA_PATH = "~/multimodal-misinformation/data/"
RESULTS_PATH = "~/multimodal-misinformation/results/"
