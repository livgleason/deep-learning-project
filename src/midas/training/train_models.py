import os
import torch
import pandas as pd

from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split

from midas.models.models import DetectionModel
from midas.data.dataset import MIDASDataset, PADDataset
from midas.data.augmentation import augment_transform, standard_transform
from midas.training.utils import *

DATA_ROOT = os.environ.get("MIDAS_DATA_ROOT")
if DATA_ROOT is None:
    raise ValueError("Set MIDAS_DATA_ROOT=/deep-learning-project/full_data")

MIDAS_IMG_DIR = os.path.join(DATA_ROOT, "MIDAS", "MIDAS_images")
PAD_IMG_DIR = os.path.join(DATA_ROOT, "PAD-UFES-20", "PAD_images")
MIDAS_CSV = os.path.join(DATA_ROOT, "MIDAS", "midas.csv")
PAD_CSV = os.path.join(DATA_ROOT, "PAD-UFES-20", "metadata.csv")


metadata = pd.read_csv(MIDAS_CSV)

train_ids, test_ids = train_test_split(metadata["midas_record_id"].unique(), test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.1765, random_state=42)

train_df = metadata[metadata["midas_record_id"].isin(train_ids)]
val_df = metadata[metadata["midas_record_id"].isin(val_ids)]
test_df = metadata[metadata["midas_record_id"].isin(test_ids)]

age_mean, age_std = compute_age_stats(train_df, "midas_age")


train_dataset = ConcatDataset([MIDASDataset(MIDAS_DIR, train_df, standard_transform, age_mean, age_std), MIDASDataset(MIDAS_DIR, train_df, augment_transform, age_mean, age_std)])
val_dataset = MIDASDataset(MIDAS_DIR, val_df, standard_transform, age_mean, age_std)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectionModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=7e-5)


best_auc = 0

for epoch in range(30):
    model.train()
    total_loss = 0

    for images, _, metadata, label in train_loader:
        images = images.squeeze(0).to(device)
        metadata = metadata.squeeze(0).to(device)
        label = label.float().to(device)

        optimizer.zero_grad()
        loss = criterion(model(images, metadata).view(1), label.view(1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    val_auc, *_ = evaluate(model, val_loader, device)

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), "model.pt")

PAD_dataset = PADDataset(PAD_IMG_DIR, PAD_CSV, standard_transform, age_mean, age_std)
PAD_loader = DataLoader(PAD_dataset, batch_size=1)
evaluate(model, PAD_loader, device)


