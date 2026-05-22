import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

data_dir = ROOT / "sample_data" / "images"
csv_file = ROOT / "sample_data" / "midas_sample.csv"

IMAGE_TYPE_MAP = {"dscope": 0,"6in": 1, "1ft": 2, "n/a - virtual": 3}
GENDER_MAP = {"female": 0, "male": 1}
LABEL_MAP = {"no": 0, "yes": 1}


class MIDASDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.patient_groups = self.data.groupby("midas_record_id")
        self.patient_ids = list(self.patient_groups.groups.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        patient_rows = self.patient_groups.get_group(patient_id)
        images = []
        image_types = []
        label = str(patient_rows["midas_melanoma"].iloc[0])
        label = LABEL_MAP.get(label, 0)
        gender = str(patient_rows["midas_gender"].iloc[0])
        gender = GENDER_MAP.get(gender, 0)
        age = float(patient_rows["midas_age"].iloc[0])
        metadata = torch.tensor([age, gender], dtype=torch.float32)
        for _, row in patient_rows.iterrows():
            image_name = row["midas_file_name"]
            image_path = os.path.join(self.data_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
            image_type = IMAGE_TYPE_MAP[row["midas_distance"]]
            image_types.append(image_type)
        images = torch.stack(images)
        image_types = torch.tensor(image_types)
        return images, image_types, metadata, label
