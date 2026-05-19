import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MIDASDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None,
                 image_col="midas_file_name", label_col="midas_melanoma"):
        self.data_dir = data_dir
        self.data = pd.read_csv(csv_file)
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_path = os.path.join(self.data_dir, row[self.image_col])
        image = Image.open(image_path).convert("RGB")
        label = row[self.label_col]
        if self.transform:
            image = self.transform(image)
        return image, label
