import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from preprocess import prepare_clip_data, MultimodalDataset
from model import ContributionAwareModel
from config import device

# === Load and prepare data ===
df_train, _, _, processor, _ = prepare_clip_data()
dataset = MultimodalDataset(df_train, processor)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# === Load trained model ===
model = ContributionAwareModel(fusion_type="cross_attention")
model.load_state_dict(torch.load("single_step_clip_model.pth", map_location=device))
model.to(device)
model.eval()

difficulty_scores = []

# === Score each training sample ===
with torch.no_grad():
    for batch in tqdm(loader, desc="Scoring training samples..."):
        inputs, labels = batch
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels = labels.to(device)

        logits, _ = model(**inputs)
        loss = F.cross_entropy(logits, labels, reduction='none')  # Scalar for each example
        difficulty_scores.extend(loss.cpu().tolist())

# === Save difficulty scores to CSV ===
df_train["difficulty_score"] = difficulty_scores
df_train.to_csv("./data/train_curriculum.csv", index=False)
print("Saved difficulty-sorted curriculum training file as train_curriculum.csv")
