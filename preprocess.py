import pandas as pd
from transformers import AutoTokenizer

# read CSV
# tokenization to CSV
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def tokenize_texts(claim, evidence):
    claim = str(claim) if pd.notna(claim) else ""
    evidence = str(evidence) if pd.notna(evidence) else ""
    
    claim_tokens = tokenizer(claim, padding='max_length', truncation=True, return_tensors="pt")
    evidence_tokens = tokenizer(evidence, padding='max_length', truncation=True, return_tensors="pt")
    
    # Remove extra batch dimension
    claim_tokens = {key: val.squeeze(0) for key, val in claim_tokens.items()}
    evidence_tokens = {key: val.squeeze(0) for key, val in evidence_tokens.items()}
    return claim_tokens, evidence_tokens