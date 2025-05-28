import pandas as pd
import torch
from model import ContributionAwareModel
from preprocess import load_and_prepare_data
from eval import two_step_inference, evaluate_model
from config import device
from sklearn.metrics import classification_report
from dataset import MultimodalDataset
from torch.utils.data import DataLoader


# Load all data
df_step1, df_step2, df_val, df_test, tokenizer, _ = load_and_prepare_data()

# print(df_step1['cleaned_truthfulness'].value_counts())
# print(df_step2['cleaned_truthfulness'].value_counts())


# # === Clean and prepare test set ===
# df_test = df_test[df_test['cleaned_truthfulness'].notna()].copy()

# # Create new test_loader to match filtered df_test
# test_dataset = MultimodalDataset(df_test)
# test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

# # === Load saved models ===
# model_step1 = ContributionAwareModel(fusion_type="cross_attention")
# model_step1.load_state_dict(torch.load("model_step1.pth"))
# model_step1.to(device)

# evaluate_model(model_step1, test_loader)

# model_step2 = ContributionAwareModel(fusion_type="cross_attention")
# model_step2.load_state_dict(torch.load("model_step2.pth"))
# model_step2.to(device)


# # === Run inference ===
# print("Running two-step inference...")
# predictions = two_step_inference(model_step1, model_step2, test_loader, device)

# # === Final Evaluation ===
# print("Final evaluation on test set (via two-step inference):")
# y_true = df_test['cleaned_truthfulness'].tolist()
# y_pred = predictions

# # Optional sanity check
# print("Length of y_true:", len(y_true))
# print("Length of y_pred:", len(y_pred))

# print(classification_report(y_true, y_pred, digits=4))