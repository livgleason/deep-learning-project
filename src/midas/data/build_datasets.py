import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from midas.data.dataloader import MIDASDataset, PADDataset, ConceptDataset
from midas.data.augmentation import augment_transform, standard_transform


DATA_ROOT = DATA_ROOT = "/gpfs/home/ogleason/deep-learning-project/full_data"

MIDAS_IMG_DIR = os.path.join(DATA_ROOT, "MIDAS", "MIDAS_images")
PAD_IMG_DIR = os.path.join(DATA_ROOT, "PAD-UFES-20", "PAD_images")
MIDAS_CSV = os.path.join(DATA_ROOT, "MIDAS", "midas.csv")
PAD_CSV = os.path.join(DATA_ROOT, "PAD-UFES-20", "metadata.csv")

MIDAS_metadata = pd.read_csv(MIDAS_CSV)
PAD_metadata = pd.read_csv(PAD_CSV)

train_ids, test_ids = train_test_split(MIDAS_metadata["midas_record_id"].unique(), test_size=0.15, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.1765, random_state=42)

train_df = MIDAS_metadata[MIDAS_metadata["midas_record_id"].isin(train_ids)]
val_df = MIDAS_metadata[MIDAS_metadata["midas_record_id"].isin(val_ids)]
test_df = MIDAS_metadata[MIDAS_metadata["midas_record_id"].isin(test_ids)]

train_dataset = ConcatDataset([MIDASDataset(MIDAS_IMG_DIR, train_df, standard_transform), MIDASDataset(MIDAS_IMG_DIR, train_df, augment_transform)])
val_dataset = MIDASDataset(MIDAS_IMG_DIR, val_df, standard_transform)
test_dataset = MIDASDataset(MIDAS_IMG_DIR, test_df, standard_transform)
PAD_dataset = PADDataset(PAD_IMG_DIR, PAD_metadata, standard_transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)
PAD_loader = DataLoader(PAD_dataset, batch_size=1)


dark_skin_df = PAD_metadata[PAD_metadata['fitspatrick'].isin([4, 5, 6])].sample(n=min(100, len(PAD_metadata[PAD_metadata['fitspatrick'].isin([4, 5, 6])])), random_state=42)
light_skin_df = PAD_metadata[PAD_metadata['fitspatrick'].isin([1, 2, 3])].sample(n=min(100, len(PAD_metadata[PAD_metadata['fitspatrick'].isin([1, 2, 3])])), random_state=42)
random_skin_df = PAD_metadata.sample(n=200, random_state=0)
random_skin_df = random_skin_df[~random_skin_df['img_id'].isin(list(dark_skin_df['img_id']) + list(light_skin_df['img_id']))].sample(n=100, random_state=0)

N = min(len(dark_skin_df), len(light_skin_df), 80)
dark_skin_df  = dark_skin_df.sample(n=N, random_state=42)
light_skin_df = light_skin_df.sample(n=N, random_state=42)
random_skin_df = random_skin_df.sample(n=N, random_state=42)

dark_skin_loader   = DataLoader(ConceptDataset(PAD_IMG_DIR, dark_skin_df, standard_transform), batch_size=16)
light_skin_loader   = DataLoader(ConceptDataset(PAD_IMG_DIR, light_skin_df, standard_transform), batch_size=16)
random_skin_loader = DataLoader(ConceptDataset(PAD_IMG_DIR, random_skin_df, standard_transform), batch_size=16)