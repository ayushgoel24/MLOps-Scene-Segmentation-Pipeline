import torch
from torch.utils.data import DataLoader, random_split
from mlops_segmentation.dataset.configuration_manager import ConfigManager

class DataLoaderWrapper:
    
    def __init__(self, dataset, config_path='dataloader_config.yaml'):
        self.dataset = dataset
        dataloader_config = ConfigManager.fetch_dataloader_config()

        self.batch_size = dataloader_config['batch_size']
        self.train_split = dataloader_config['train_split']
        self.val_split = dataloader_config['val_split']
        self.shuffle = dataloader_config['shuffle']
        self.num_workers = dataloader_config['num_workers']

        self.train_loader, self.val_loader, self.test_loader = self.create_loaders()
    
    @staticmethod
    def calculate_split_sizes(dataset_size, train_split, val_split):
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        return train_size, val_size, test_size

    def create_loaders(self):
        total_size = len(self.dataset)
        train_size, val_size, test_size = self.calculate_split_sizes(total_size, self.train_split, self.val_split)

        if self.shuffle:
            train_dataset, val_dataset, test_dataset = random_split(self.dataset, [train_size, val_size, test_size])
        else:
            train_dataset = torch.utils.data.Subset(self.dataset, range(0, train_size))
            val_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + val_size))
            test_dataset = torch.utils.data.Subset(self.dataset, range(train_size + val_size, total_size))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
