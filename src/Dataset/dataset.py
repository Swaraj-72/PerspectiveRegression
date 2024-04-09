import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        image = Image.open("artifacts/imgs/" + img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        output = np.array(self.data.iloc[idx, 1:], dtype=np.float32)
        return image, output