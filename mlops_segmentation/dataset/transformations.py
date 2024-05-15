from torchvision import transforms

class Transformations:
    
    @staticmethod
    def basic_transform():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def augment_transform():
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def resize_transform():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    @staticmethod
    def to_tensor_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])
    
    @staticmethod
    def cityscape_transform():
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.56, 0.406), 
                std=(0.229, 0.224, 0.225)
            )
        ])
    
    @staticmethod
    def cityscape_inverse_transform():
        return transforms.Compose([
            transforms.Normalize(
                mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225), 
                std=(1/0.229, 1/0.224, 1/0.225)
            )
        ])

    @staticmethod
    def get_transform(transform_name):
        # Map transform names to static methods
        transforms_map = {
            'basic': Transformations.basic_transform(),
            'augmentation': Transformations.augment_transform(),
            'resize_only': Transformations.resize_transform(),
            'to_tensor_only': Transformations.to_tensor_transform(),
            'cityscapes': Transformations.cityscape_transform(),
            'cityscapes_inv': Transformations.cityscape_inverse_transform
        }
        return transforms_map.get(transform_name, None)