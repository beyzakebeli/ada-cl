import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def evaluate_model(model, test_loader, save_path='./results/adaptive_CL_with_modality_contribution_NEI.txt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)  # make sure model is on the cuda device
    model.eval() # evaluation mode

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            text, evidence, image, labels = batch

            # Move to GPU
            text = {key: val.to(device) for key, val in text.items()}
            evidence = {key: val.to(device) for key, val in evidence.items()}
            image = image.to(device)
            labels = labels.to(device)

            # Forward pass (model prediction)
            logits, _ = model(text=text, evidence=evidence, image=image)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())  # Store predictions
            all_labels.extend(labels.cpu().numpy())  # Store labels

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics computation
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds)

    # Print results
    print("\n **Test Set Performance**")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\n Classification Report:\n", classification_report(all_labels, all_preds))

    # Saving the results
    with open(save_path, "a", encoding="utf-8") as f:
        f.write("**Test Set Performance**\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    return acc, precision, recall, f1 

def two_step_inference(model_step1, model_step2, dataloader, device):
    model_step1.eval()
    model_step2.eval()
    all_preds = []

    with torch.no_grad():
        for text, evidence, image, _ in dataloader:
            text = {k: v.to(device) for k, v in text.items()}
            evidence = {k: v.to(device) for k, v in evidence.items()}
            image = image.to(device)

            step1_logits, _ = model_step1(text=text, evidence=evidence, image=image)
            step1_pred = step1_logits.argmax(dim=1)

            for i, pred in enumerate(step1_pred):
                if pred.item() == 0:  # NEI
                    all_preds.append(2)
                else:
                    single_text = {k: v[i].unsqueeze(0) for k, v in text.items()}
                    single_evidence = {k: v[i].unsqueeze(0) for k, v in evidence.items()}
                    single_image = image[i].unsqueeze(0)

                    step2_logits, _ = model_step2(text=single_text, evidence=single_evidence, image=single_image)
                    step2_pred = step2_logits.argmax(dim=1).item()
                    all_preds.append(step2_pred)  # 0 or 1