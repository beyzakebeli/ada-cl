from preprocess import load_and_prepare_data
from model import ContributionAwareModel
from train import train_step1_model
from eval import evaluate_model
from config import device

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("Preparing data...")
df_step1, df_step2, df_val, df_test, tokenizer, test_loader = load_and_prepare_data()

model_step1 = ContributionAwareModel()
optimizer1 = optim.AdamW(model_step1.parameters(), lr=2e-5, weight_decay=1e-2)
scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', patience=3, factor=0.5)

print("Starting Step 1 Training...")
train_step1_model(
    model=model_step1,
    df_step1=df_step1,
    test_loader=test_loader,
    tokenizer=tokenizer,
    device=device,
    optimizer=optimizer1,
    scheduler=scheduler1,
    epochs=5,
    batch_size=4,
    max_images=5
)
