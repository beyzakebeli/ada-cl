import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from model import ContributionAwareModel
from preprocess import load_and_prepare_data
from config import device
from dataset import MultimodalDataset

# === Load data and tokenizer ===
df_step1, df_step2, df_val, df_test, tokenizer, _ = load_and_prepare_data()

# === Prepare binary labels for Step 1 ===
df_test = df_test[df_test['cleaned_truthfulness'].notna()].copy()
df_test['binary_label_step1'] = df_test['cleaned_truthfulness'].apply(lambda x: 0 if x == 2 else 1)

# === Load test set for step 1 ===
test_dataset = MultimodalDataset(df_test)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

# === Load trained Step 1 model ===
model_step1 = ContributionAwareModel(fusion_type="cross_attention")
model_step1.load_state_dict(torch.load("model_step1.pth"))
model_step1.to(device)
model_step1.eval()

# === Inference (Binary) ===
all_preds = []
with torch.no_grad():
    for text, evidence, image, _ in test_loader:
        text = {k: v.to(device) for k, v in text.items()}
        evidence = {k: v.to(device) for k, v in evidence.items()}
        image = image.to(device)

        logits, _ = model_step1(text=text, evidence=evidence, image=image)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)

# === Evaluation ===
y_true = df_test['binary_label_step1'].tolist()
y_pred = all_preds

print("Step 1 Binary Evaluation (NEI vs Non-NEI):")
print(classification_report(y_true, y_pred, target_names=["NEI", "Refuted/Supported"], digits=4))
