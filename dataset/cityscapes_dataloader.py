import torch
from torch.utils.data import DataLoader, random_split
from dataset.configuration_manager import ConfigManager
from dataset.cityscapes_dataset import CityScapesDataset
import os


class CityScapes_DataLoader:
    
    def __init__(self):
        
        dataset_config = ConfigManager.fetch_dataset_config()
        dataloader_config = ConfigManager.fetch_dataloader_config()

        batch_size = dataloader_config['batch_size']
        train_split = dataloader_config['train_split']

        train_validation_dataset = CityScapesDataset(
            image_dir=os.path.join(dataset_config['image_dir'], 'train'), 
            label_dir=os.path.join(dataset_config['label_dir'], 'train'),
            transform=dataset_config['transform']
            )
        
        num_train = len(train_validation_dataset)
        train_size = round(num_train * train_split)
        VAL_SET_SIZE = num_train - train_split

        train_set, val_set = torch.utils.data.random_split(
            train_validation_dataset,
            [train_size, VAL_SET_SIZE],
            generator=torch.Generator().manual_seed(1)
        )

        test_dataset = CityScapesDataset(
            image_dir=os.path.join(dataset_config['image_dir'], 'val'), 
            label_dir=os.path.join(dataset_config['label_dir'], 'val'),
            transform=dataset_config['transform']
            )
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader