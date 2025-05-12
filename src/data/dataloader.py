import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __getitem__(self, idx):
        pixels = np.array(self.df.iloc[idx]['pixels'].split(), dtype='float32').reshape(48, 48)
        img = torch.tensor(pixels).unsqueeze(0)  # Add channel dim
        label = int(self.df.iloc[idx]['emotion'])
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)