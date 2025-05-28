from preprocess import load_and_prepare_data
from model import ContributionAwareModel
from train import train_step1_model, train_step2_with_curriculum
from eval import evaluate_model, two_step_inference
from config import device
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report


# Optional memory setting
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

print("Preparing data...")
df_step1, df_step2, df_val, df_test, tokenizer, test_loader = load_and_prepare_data()

# Fix label column for binary setting
df_step1['cleaned_truthfulness'] = df_step1['binary_label']
df_step2['cleaned_truthfulness'] = df_step2['binary_label']

# === STEP 1 ===
model_step1 = ContributionAwareModel()
model_step1 = ContributionAwareModel(fusion_type="cross_attention")
optimizer1 = optim.AdamW(model_step1.parameters(), lr=2e-5, weight_decay=1e-2)
scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', patience=3, factor=0.5)
print('Moving model to CUDA...')
model_step1.to(device)

print("Starting Step 1 Training...")
train_step1_model(
    model=model_step1,
    df_step1=df_step1,
    test_loader=test_loader,
    tokenizer=tokenizer,
    device=device,
    optimizer=optimizer1,
    scheduler=scheduler1,
    epochs=3,
    batch_size=1,
    max_images=3
)
torch.save(model_step1.state_dict(), "model_step1.pth")
print("Trained step 1 model is saved!")
# model_step1 = ContributionAwareModel(fusion_type="cross_attention")
# model_step1.load_state_dict(torch.load("model_step1.pth"))
# model_step1.to(device)

# === STEP 2 ===
# model_step2 = ContributionAwareModel()
model_step2 = ContributionAwareModel(fusion_type="cross_attention")
optimizer2 = optim.AdamW(model_step2.parameters(), lr=2e-5, weight_decay=1e-2)
scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', patience=3, factor=0.5)

print("Starting Step 2 Training (Refuted vs Supported)...")
train_step2_with_curriculum(
    model=model_step2,
    df_step2=df_step2,
    test_loader=test_loader,
    tokenizer=tokenizer,
    device=device,
    optimizer=optimizer2,
    scheduler=scheduler2,
    epochs=15,
    batch_size=1,
    max_images=3,
    warmup_ratio=0.3,
    pacing_increase=0.05,
    lambda_kl=1.0,
    alpha=0.1
)
torch.save(model_step2.state_dict(), "model_step2.pth")

# === TWO-STEP INFERENCE ===
print("Running inference with two-step model...")
predictions = two_step_inference(
    model_step1=model_step1,
    model_step2=model_step2,
    dataloader=test_loader,
    device=device
)

# Add predictions to test DataFrame (optional)
df_test["predicted_label"] = predictions

# === EVALUATION ===
print("Final evaluation on test set (via two-step inference):")
df_test = df_test[df_test['cleaned_truthfulness'].notna()]  # Ensure labels are valid
y_true = df_test['cleaned_truthfulness'].tolist()
y_pred = predictions

print(classification_report(y_true, y_pred, digits=4))
