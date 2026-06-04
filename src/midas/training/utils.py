import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def compute_age_stats(df, column):
    ages = df[column].values.astype(float)
    mean = np.mean(ages)
    std = np.std(ages)
    return mean, std

def find_best_threshold(y_true, probs):
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def get_probs_and_labels(model, loader, device):
    model.eval()
    probs, labels = [], []
    
    with torch.no_grad():
        for images, _, metadata, label in loader:
            images = images.squeeze(0).to(device)
            metadata = metadata.squeeze(0).to(device)

            logits = model(images, metadata)
            prob = torch.sigmoid(logits).view(-1)

            probs.append(prob.cpu())
            labels.append(label.view(-1).cpu())
            
    return torch.cat(labels).numpy(), torch.cat(probs).numpy()

def evaluate(model, loader, device):
    y_true, probs = get_probs_and_labels(model, loader, device)
    auc = roc_auc_score(y_true, probs)

    thresh, f1 = find_best_threshold(y_true, probs)
    preds = (probs >= thresh).astype(int)

    return (auc, accuracy_score(y_true, preds), recall_score(y_true, preds), f1_score(y_true, preds), thresh)