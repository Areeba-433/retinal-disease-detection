import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def get_transforms(phase, image_size=224):
    if phase == "train":
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


class ODIRDataset(Dataset):
    def __init__(self, df, image_dir, phase="train", image_size=224):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.alt_dir = image_dir.replace("ODIR-5K", "preprocessed_images")
        self.transforms = get_transforms(phase, image_size)
        self.label_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        image = cv2.imread(img_path)

        if image is None:
            alt_path = os.path.join(self.alt_dir, row["filename"])
            image = cv2.imread(alt_path)

        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = apply_clahe(image)

        augmented = self.transforms(image=image)
        image = augmented["image"]

        label = torch.tensor(
            row[self.label_cols].values.astype(float),
            dtype=torch.float32
        )
        return image, label