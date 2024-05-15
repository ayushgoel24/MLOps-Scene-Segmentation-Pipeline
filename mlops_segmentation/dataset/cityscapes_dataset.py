import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from cityscapesscripts.helpers.labels import trainId2label

class CityscapesDataset(Dataset):

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = [f for f in os.listdir(images_dir)]
        self.masks = [f for f in os.listdir(masks_dir)]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        mask = np.array(mask)

        mask = np.where(mask == -1, 255, mask)
        mask = self.encode_segmap(mask)

        mask = torch.tensor(mask, dtype=torch.long)
        return image, mask
    
    @staticmethod
    def encode_segmap(mask):
        mask = np.squeeze(mask)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for train_id, label in trainId2label.items():
            label_mask[mask == label.id] = train_id
        return label_mask