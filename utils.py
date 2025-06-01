import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch
from dataset import MultimodalDataset
from torch.utils.data import DataLoader


def merge_evidence(df):
    return df.groupby("Claim", as_index=False).agg({
        "Evidence": lambda x: " [SEP] ".join(set(x.dropna())) if x.dropna().any() else "",
        "image_path": "first",  # Keep first image list
        "cleaned_truthfulness": "first"  # Keep first label
    })

def prepare_two_step_datasets(df):
    df_step1 = df.copy()
    df_step1['binary_label'] = df_step1['cleaned_truthfulness'].apply(lambda x: 0 if x == 2 else 1)  # 0 = NEI, 1 = Refuted or Supported

    df_step2 = df[df['cleaned_truthfulness'] != 2].copy()
    df_step2['binary_label'] = df_step2['cleaned_truthfulness'].apply(lambda x: 0 if x == 0 else 1)  # 0 = Refuted, 1 = Supported

    return df_step1, df_step2

# tokenizer is changed with processor for CLIP-only scenario
def initialize_difficulty_scores(df, model, processor, device):
    model.eval()
    scores = []
    # dataset = MultimodalDataset(df)
    dataset = MultimodalDataset(df, processor=processor) # for CLIP
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            # text, evidence, images, labels = batch
            # text = {k: v.to(device) for k, v in text.items()}
            # evidence = {k: v.to(device) for k, v in evidence.items()}
            # images, labels = images.to(device), labels.to(device)
            # logits, _ = model(text=text, evidence=evidence, image=images)
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            logits, _ = model(**inputs)
            losses = F.cross_entropy(logits, labels, reduction='none')
            scores.extend(losses.cpu().tolist())
    df['difficulty_score'] = scores
    return df

def update_difficulty_scores(df, new_losses, alpha=0.1):
    df['difficulty_score'] = (1 - alpha) * df['difficulty_score'] + alpha * np.array(new_losses)
    return df

def kl_divergence_with_logits(p_logits, q_logits):
    p_log_prob = F.log_softmax(p_logits, dim=-1)
    q_prob = F.softmax(q_logits, dim=-1)
    kl = F.kl_div(p_log_prob, q_prob, reduction='batchmean')
    return kl

def get_easy_examples_per_class(df, pacing_ratio=0.2, max_per_class=500):
    easy_samples = []
    for label in df['cleaned_truthfulness'].unique():
        class_subset = df[df['cleaned_truthfulness'] == label]
        class_subset = class_subset.sort_values("difficulty_score", ascending=True)
        n_samples = int(len(class_subset) * pacing_ratio)
        n_samples = min(n_samples, max_per_class)
        easy_samples.append(class_subset.head(n_samples))
    return pd.concat(easy_samples)


