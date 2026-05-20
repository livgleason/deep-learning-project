import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

data_dir = ROOT / "sample_data" / "images"
csv_file = ROOT / "sample_data" / "midas_sample.csv"

class MIDASDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        self.patient_groups = self.data.groupby("midas_record_id")
        self.patient_ids = list(self.patient_groups.groups.keys())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient_id = self.patient_ids[index]
        patient_rows = self.patient_groups.get_group(patient_id)
        images = []
        image_types = []
        label = patient_rows["midas_melanoma"].iloc[0]
        for _, row in patient_rows.iterrows():
            image_name = row["midas_file_name"]
            image_path = os.path.join(self.data_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
            image_type = row["midas_distance"]
            image_types.append(image_type)
        images = torch.stack(images)
        image_types = torch.tensor(image_types)
        return images, image_types, label
