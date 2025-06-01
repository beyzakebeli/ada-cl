import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from llava.processor import LlavaProcessor  # From HuggingFace's llava-hf

# === CONFIG ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

MODEL_NAME = "llava-hf/llava-v1.6-mistral-7b-hf"
TRAIN_FILE = "./data/train_curriculum.csv"
OUTPUT_DIR = "./checkpoints/llava_curriculum_peft"
EPOCHS = 15
BATCH_SIZE = 1
ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP_RATIO = 0.3
PACING_INCREASE = 0.1
MAX_LENGTH = 512

class CurriculumDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        prompt = f"Claim: {row['Claim']}\nEvidence: {row['Evidence']}\nImage: {row['image_path']}\nAnswer:"
        inputs = self.processor(
            prompt,
            row['image_path'],
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH
        )
        inputs['labels'] = self.processor.tokenizer(
            row['cleaned_truthfulness'],
            return_tensors='pt',
            truncation=True,
            max_length=MAX_LENGTH
        ).input_ids[0]
        return {k: v.squeeze(0) for k, v in inputs.items()}

def get_curriculum_subset(df, pacing_ratio):
    selected_samples = []
    for label in df['cleaned_truthfulness'].unique():
        class_subset = df[df['cleaned_truthfulness'] == label]
        class_subset = class_subset.sort_values("difficulty_score")
        pacing_samples = int(len(class_subset) * pacing_ratio)
        selected = class_subset.iloc[:pacing_samples]
        selected_samples.append(selected)
    return pd.concat(selected_samples).sample(frac=1).reset_index(drop=True)

def main():
    processor = LlavaProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model = model.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    df_full = pd.read_csv(TRAIN_FILE)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        pacing_ratio = min(WARMUP_RATIO + (epoch - 1) * PACING_INCREASE, 1.0)
        df_subset = get_curriculum_subset(df_full, pacing_ratio)

        dataset = CurriculumDataset(df_subset, processor)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(loader, desc="Training")):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            epoch_loss += loss.item() * ACCUMULATION_STEPS

        print(f"Epoch {epoch} loss: {epoch_loss:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
