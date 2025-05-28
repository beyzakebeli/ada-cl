import pandas as pd
import ast
from transformers import AutoTokenizer
from dataset import MultimodalDataset
from torch.utils.data import DataLoader

from utils import merge_evidence, prepare_two_step_datasets

def tokenize_texts(claim, evidence, tokenizer):
    claim = str(claim) if pd.notna(claim) else ""
    evidence = str(evidence) if pd.notna(evidence) else ""

    # max_length=128'ler sonradan eklendi
    claim_tokens = tokenizer(claim, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    evidence_tokens = tokenizer(evidence, padding='max_length', truncation=True, max_length=128, return_tensors="pt")

    # Remove batch dimension
    claim_tokens = {k: v.squeeze(0) for k, v in claim_tokens.items()}
    evidence_tokens = {k: v.squeeze(0) for k, v in evidence_tokens.items()}
    return claim_tokens, evidence_tokens

def load_and_prepare_data():
    # Load datasets
    df_train = pd.read_csv("~/multimodal-misinformation/data/train_mocheg.csv")
    df_val = pd.read_csv("~/multimodal-misinformation/data/val_mocheg.csv")
    df_test = pd.read_csv("~/multimodal-misinformation/data/test_mocheg.csv")

    # Drop difficulty scores if already present
    for df in [df_train, df_val, df_test]:
        df.drop(columns=['difficulty_score'], errors='ignore', inplace=True)

    # Convert image_path from string to list
    for df in [df_train, df_val, df_test]:
        df["image_path"] = df["image_path"].apply(ast.literal_eval)

    # Merge evidence per claim
    df_train = merge_evidence(df_train)
    df_val = merge_evidence(df_val)
    df_test = merge_evidence(df_test)

    # Drop NaNs and map labels
    label_map = {'refuted': 0, 'supported': 1, 'NEI': 2}
    for df in [df_train, df_val, df_test]:
        df.dropna(subset=['cleaned_truthfulness'], inplace=True)
        df['cleaned_truthfulness'] = df['cleaned_truthfulness'].map(label_map)

    # Split train into step 1 (3-class) and step 2 (2-class)
    df_step1, df_step2 = prepare_two_step_datasets(df_train)

    # Load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    # yukaridaki use_fast napiyor bilmiyorum

    # Tokenize
    for df in [df_step1, df_step2, df_val, df_test]:
        df['tokenized_claim'], df['tokenized_evidence'] = zip(*df.apply(
            lambda row: tokenize_texts(row["Claim"], row["Evidence"], tokenizer), axis=1))

    # Create test dataloader
    test_dataset = MultimodalDataset(df_test)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)

    df_step1['cleaned_truthfulness'] = df_step1['binary_label']
    df_step2['cleaned_truthfulness'] = df_step2['binary_label']

    return df_step1, df_step2, df_val, df_test, tokenizer, test_loader
