import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from dataset import MultimodalDataset
from eval import evaluate_model
from utils import kl_divergence_with_logits, initialize_difficulty_scores
from collections import Counter
from tqdm import tqdm

def train_baseline_model(
    model,
    df_train,
    test_loader,
    tokenizer,
    device,
    optimizer,
    scheduler,
    epochs=5,
    batch_size=4,
    max_images=3,
    use_class_weights=False
):

    train_dataset = MultimodalDataset(df_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            text, evidence, images, labels = batch
            text = {k: v.to(device) for k, v in text.items()}
            evidence = {k: v.to(device) for k, v in evidence.items()}
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(text=text, evidence=evidence, image=images)

            if use_class_weights:
                class_weights = torch.tensor([1.0, 1.0, 2.0], device=device)
                loss = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step(total_loss / len(train_loader))

        print(f"\nEpoch {epoch+1} Evaluation:")
        evaluate_model(model, test_loader)

    return model

# def train_with_adaptive_difficulty(model, df_train, test_loader, tokenizer, device, optimizer, scheduler,
#                                    epochs=10, batch_size=2, max_images=5,
#                                    warmup_ratio=0.3, pacing_increase=0.1, lambda_kl=1.0, alpha=-0.1):
#     model.to(device)
#     scaler = GradScaler()

#     result_path = './results/cross-modal-attention.txt'
#     os.makedirs(os.path.dirname(result_path), exist_ok=True)
#     open(result_path, 'w').close()

#     for epoch in range(1, epochs + 1):
#         print(f"\n=== Epoch {epoch}/{epochs} ===")
#         model.train()

#         if epoch == 1:
#             print("Initializing difficulty scores on full training set...")
#             df_train = initialize_difficulty_scores(df_train, model, tokenizer, device)

#         pacing_ratio = min(warmup_ratio + (epoch - 1) * pacing_increase, 1.0)

#         selected_samples = []
#         for label in df_train['cleaned_truthfulness'].unique():
#             class_subset = df_train[df_train['cleaned_truthfulness'] == label]
#             class_subset = class_subset.sort_values("difficulty_score")
#             pacing_samples = int(len(class_subset) * pacing_ratio)
#             selected = class_subset.iloc[:pacing_samples]
#             selected_samples.append(selected)

#         train_subset = pd.concat(selected_samples)
#         print(f"Training on {len(train_subset)} samples (Pacing Ratio: {pacing_ratio:.2f})")
#         print("Class distribution in selected samples:")
#         print(train_subset['cleaned_truthfulness'].value_counts())

#         train_loader = DataLoader(MultimodalDataset(train_subset, max_images=max_images),
#                                   batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

#         epoch_losses = []
#         optimizer.zero_grad()
        
#         for batch_idx, (text, evidence, image, labels) in enumerate(train_loader):
#             text = {k: v.to(device) for k, v in text.items()}
#             evidence = {k: v.to(device) for k, v in evidence.items()}
#             image = image.to(device)
#             labels = labels.to(device)

#             with autocast():
#                 logits, contrib_scores = model(text=text, evidence=evidence, image=image)
#                 loss_xe = F.cross_entropy(logits, labels)
#                 loss_kl = kl_divergence_with_logits(logits, logits.detach())

#                 # Contribution loss: ters loss üzerinden pseudo score üret
#                 with torch.no_grad():
#                     confidence = torch.softmax(logits, dim=-1)
#                     pred_prob = confidence[torch.arange(len(labels)), labels]  # B
#                     pseudo_contrib = 1.0 - pred_prob.unsqueeze(1).repeat(1, 3)  # lower prob → higher loss → higher supervision

#                 pseudo_contrib = pseudo_contrib / (pseudo_contrib.sum(dim=1, keepdim=True) + 1e-6)
#                 loss_contrib = compute_contribution_loss(logits, labels, contrib_scores)

#                 loss = loss_xe + lambda_kl * loss_kl + loss_contrib

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             optimizer.zero_grad()

#             batch_losses = F.cross_entropy(logits, labels, reduction='none')
#             epoch_losses.extend(batch_losses.detach().cpu().tolist())

#             torch.cuda.empty_cache()

#         # ONLY update difficulty_score of relevant indices
#         df_train.loc[train_subset.index, 'difficulty_score'] = (
#             (1 - alpha) * df_train.loc[train_subset.index, 'difficulty_score'] + alpha * np.array(epoch_losses)
#         )

#         scheduler.step(np.mean(epoch_losses))

#         print(f"\nEvaluating after Epoch {epoch}...")
#         evaluate_model(model, test_loader, save_path=result_path)

def train_step1_model(model, df_step1, test_loader, tokenizer, device, optimizer, scheduler,
                      epochs=5, batch_size=4, max_images=5):
    model.to(device)
    scaler = GradScaler()
    model.train()

    # === Compute class weights ===
    label_counts = Counter(df_step1['binary_label'].tolist())  # 0 = NEI, 1 = Refuted/Supported
    total = sum(label_counts.values())
    class_weights = [total / label_counts[cls] for cls in range(len(label_counts))]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Create weighted loss function
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Prepare data loader
    dataset = MultimodalDataset(df_step1, max_images=max_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for text, evidence, image, labels in loader:
            text = {k: v.to(device) for k, v in text.items()}
            evidence = {k: v.to(device) for k, v in evidence.items()}
            image = image.to(device)
            labels = labels.to(device)

            with autocast():
                logits, _ = model(text=text, evidence=evidence, image=image)
                loss = loss_fn(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())

        print(f"[Step 1] Epoch {epoch+1} | Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")

def train_step2_with_curriculum(model, df_step2, test_loader, tokenizer, device, optimizer, scheduler,
                                epochs=10, batch_size=4, max_images=5,
                                warmup_ratio=0.3, pacing_increase=0.05, lambda_kl=1.0, alpha=0.1):
    train_with_adaptive_difficulty(
        model=model,
        df_train=df_step2,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        batch_size=batch_size,
        max_images=max_images,
        warmup_ratio=warmup_ratio,
        pacing_increase=pacing_increase,
        lambda_kl=lambda_kl,
        alpha=alpha
    )

def compute_contribution_loss(preds, targets, contrib_scores, gamma=1e-3):
    confidence = torch.softmax(preds, dim=-1)
    true_probs = confidence[torch.arange(len(targets)), targets]  # [B]
    pseudo_scores = 1.0 - true_probs.unsqueeze(1).repeat(1, 2)     # [B, 3] changed into [B, 2] because for CLIP, claim and text evidence are taken together!
    pseudo_scores = pseudo_scores / (pseudo_scores.sum(dim=1, keepdim=True) + 1e-6)

    contrib_loss = F.l1_loss(contrib_scores, pseudo_scores.detach())
    return contrib_loss * gamma

def train_with_adaptive_difficulty(model, df_train, test_loader, processor, device, optimizer, scheduler,
                                   epochs=10, batch_size=2, max_images=5,
                                   warmup_ratio=0.3, pacing_increase=0.1, lambda_kl=1.0, alpha=-0.1):
    model.to(device)
    scaler = GradScaler()

    result_path = './results/cross-modal-attention-clip.txt'
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    open(result_path, 'w').close()

    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        model.train()

        if epoch == 1:
            print("Initializing difficulty scores on full training set...")
            df_train = initialize_difficulty_scores(df_train, model, processor, device)

        pacing_ratio = min(warmup_ratio + (epoch - 1) * pacing_increase, 1.0)

        selected_samples = []
        for label in df_train['cleaned_truthfulness'].unique():
            class_subset = df_train[df_train['cleaned_truthfulness'] == label]
            class_subset = class_subset.sort_values("difficulty_score")
            pacing_samples = int(len(class_subset) * pacing_ratio)
            selected = class_subset.iloc[:pacing_samples]
            selected_samples.append(selected)

        train_subset = pd.concat(selected_samples)
        print(f"Training on {len(train_subset)} samples (Pacing Ratio: {pacing_ratio:.2f})")
        print("Class distribution in selected samples:")
        print(train_subset['cleaned_truthfulness'].value_counts())

        train_loader = DataLoader(MultimodalDataset(train_subset, processor=processor, max_images=max_images),
                                  batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

        epoch_losses = []
        optimizer.zero_grad()

        for batch_idx, (clip_inputs, labels) in enumerate(train_loader):
            clip_inputs = {k: v.to(device) for k, v in clip_inputs.items()}
            labels = labels.to(device)

            with autocast():
                logits, contrib_scores = model(**clip_inputs)
                loss_xe = F.cross_entropy(logits, labels)
                loss_kl = kl_divergence_with_logits(logits, logits.detach())

                with torch.no_grad():
                    confidence = torch.softmax(logits, dim=-1)
                    pred_prob = confidence[torch.arange(len(labels)), labels]
                    pseudo_contrib = 1.0 - pred_prob.unsqueeze(1).repeat(1, 2)
                    pseudo_contrib = pseudo_contrib / (pseudo_contrib.sum(dim=1, keepdim=True) + 1e-6)

                loss_contrib = compute_contribution_loss(logits, labels, contrib_scores)
                loss = loss_xe + lambda_kl * loss_kl + loss_contrib

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            batch_losses = F.cross_entropy(logits, labels, reduction='none')
            epoch_losses.extend(batch_losses.detach().cpu().tolist())

            torch.cuda.empty_cache()

        df_train.loc[train_subset.index, 'difficulty_score'] = (
            (1 - alpha) * df_train.loc[train_subset.index, 'difficulty_score'] + alpha * np.array(epoch_losses)
        )

        scheduler.step(np.mean(epoch_losses))

        print(f"\nEvaluating after Epoch {epoch}...")
        evaluate_model(model, test_loader, save_path=result_path)
