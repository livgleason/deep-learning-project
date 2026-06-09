import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from midas.models.models import DetectionModel
print("✅ train_models.py file loaded")

def main():
    print("Starting training script...")
    from midas.data.build_datasets import train_loader, val_loader

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DetectionModel().to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    for param in model.encoder.parameters():
        param.requires_grad = False
    for name, param in model.encoder.named_parameters():
        if "layer2" in name or "layer3" in name or "layer4" in name:
            param.requires_grad = True

    num_epochs = 25
    early_stop_patience = 5
    epochs_no_improve = 0

    train_losses = []
    val_losses = []
    val_aucs = []
    best_auc = 0

    print("Beginning training loop...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, _, _, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.squeeze(0).to(device)
            label = label.float().to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits.view(1), label.view(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        all_probs = []
        all_labels = []
        val_loss_total = 0

        with torch.no_grad():
            for images, _, _, label in val_loader:
                images = images.squeeze(0).to(device)
                label = label.float().to(device)

                logits = model(images)
                loss = criterion(logits.view(1), label.view(1))
                val_loss_total += loss.item()

                probs = torch.sigmoid(logits)
                all_probs.append(probs.view(-1).cpu())
                all_labels.append(label.view(-1).cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        val_loss = val_loss_total / len(val_loader)
        val_losses.append(val_loss)

        val_auc = roc_auc_score(all_labels, all_probs)
        val_aucs.append(val_auc)

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"\nEpoch {epoch+1}/{num_epochs}: "f"Train Loss: {avg_train_loss:.4f}, "f"Val Loss: {val_loss:.4f}, "f"Val AUC: {val_auc:.4f}")
    print("Training complete.")

if __name__ == "__main__":
    main()