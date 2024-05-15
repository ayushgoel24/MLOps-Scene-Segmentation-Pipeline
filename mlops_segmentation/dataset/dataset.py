import cv2
import torch
from torch.utils.data import Dataset
import os
from mlops_segmentation.dataset.transformations import Transformations

class BaseDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            label_dir (string): Directory with all the labels.
            dataset_type (string): Type of dataset (e.g., 'cityscapes', 'nuscenes', 'ade20k').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = Transformations.get_transform(transform)
        
        self.images = sorted(os.listdir(image_dir))
        self.labels = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        sourceImage = cv2.imread(img_path, -1)
        sourceImage = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2RGB)
        if self.transform:
            sourceImage = self.transform(sourceImage)

        label_path = os.path.join(self.label_dir, self.labels[idx])
        labelImage = cv2.imread(label_path, -1)
        labelImage = torch.from_numpy(labelImage).long()

        return sourceImage, labelImage