from cityscapesscripts.helpers.labels import trainId2label
import mlflow
import mlflow.pytorch
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import segmentation_models_pytorch as smp
from sklearn.metrics import jaccard_score
import wandb

from logger.manager import LoggerManager
from dataset.cityscapes_dataloader import CityScapes_DataLoader
from mean_iou import meanIoU
from models.custom_model import FCN_resnet50
from prune.model_prune import ModelPruner
from quantize.model_quant import ModelQuantization

NUM_CLASSES = len(trainId2label)

def calculate_mean_iou(outputs, targets, num_classes):
    outputs = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    ious = []
    for cls in range(num_classes):
        iou = jaccard_score(targets.flatten(), outputs.flatten(), average=None, labels=[cls])
        ious.append(iou[0])

    mean_iou = np.mean(ious)
    return mean_iou


class CustomTrainer:
    
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)
        self.logger = LoggerManager(log_file_name="custom_trainer.log", logger_name="custom_trainer").get_logger()

    def validate(self, criterion):
        self.model.eval()
        total_loss = 0.0
        metric_object = meanIoU(NUM_CLASSES)

        with torch.no_grad():
            for inputs, labels in tqdm(self.test_loader, total=len(self.test_loader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                y_preds = self.model(inputs)

                # calculate loss
                loss = criterion(y_preds, labels)
                total_loss += loss.item()
           
                metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())

        evaluation_loss = total_loss / len(self.test_loader)
        evaluation_metric = metric_object.compute()
        return evaluation_loss, evaluation_metric

    def train(self, optimizer, criterion, config=None):
        with wandb.init(config=config):
            config = wandb.config
            num_epochs = config.epochs

            # Start MLflow run
            mlflow.start_run()
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("batch_size", config.batch_size)
            mlflow.log_param("learning_rate", config.learning_rate)

            for epoch in range(num_epochs):
                print(f"Running epoch {epoch+1}/{num_epochs}")
                self.model.train()
                running_loss = 0.0
                running_iou = 0.0
                with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                    for images, masks in self.train_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()

                        mean_iou = calculate_mean_iou(outputs, masks, len(trainId2label))
                        running_iou += mean_iou

                        pbar.update(1)
                        pbar.set_postfix(loss=loss.item(), mean_iou=mean_iou)

                avg_loss = running_loss / len(self.train_loader)
                avg_iou = running_iou / len(self.train_loader)

                validation_loss, validation_metric = self.validate(criterion)

                wandb.log({"epoch": epoch, "loss": avg_loss, "mean_iou": avg_iou, "val_loss": validation_loss, "val_iou": validation_metric})
                mlflow.log_metric("loss", avg_loss, step=epoch)
                mlflow.log_metric("mean_iou", avg_iou, step=epoch)
                mlflow.log_metric("val_loss", validation_loss, step=epoch)
                mlflow.log_metric("val_iou", validation_metric, step=epoch)

                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}, Mean IoU: {avg_iou}")

            print("Finished Training")

            # Log the trained model to W&B and MLflow
            torch.save(self.model.state_dict(), "model_weights/custom_cityscapes.pth")
            wandb.save("custom_cityscapes.pth")
            mlflow.pytorch.log_model(self.model, "model")

            mlflow.end_run()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))


class CustomTrainWrapper:

    @staticmethod
    def run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = FCN_resnet50(num_classes=NUM_CLASSES)
        train_loader, val_loader, test_loader = CityScapes_DataLoader()

        criterion = smp.losses.DiceLoss('multiclass', classes=[0,1,2], log_loss=True, smooth=1.0)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        scheduler = OneCycleLR(optimizer, max_lr=3e-4, epochs=10, steps_per_epoch=len(train_loader), pct_start=0.3, div_factor=10, anneal_strategy='cos')
        mlflow.set_experiment("Cityscapes_Semantic_Segmentation_Custom")

        customTrainer = CustomTrainer(model=model, train_loader=train_loader, test_loader=val_loader, device=device)
        customTrainer.train(optimizer=optimizer, criterion=criterion)

        pruner = ModelPruner(model=model, device=device, test_loader=test_loader)
        _, pruned_model = pruner.prune_model()

        quantized_model = ModelQuantization.run(model=pruned_model)
        torch.jit.save( torch.jit.script(quantized_model), 'model_weights/quantized_custom_model.pt' )


class CustomInferer:
    
    def __init__(self, model_path, num_classes, device=None):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.Resize((256, 512)),
            transforms.ToTensor(),
        ])

    def load_model(self):
        model = FCN_resnet50(num_classes=NUM_CLASSES)
        model = torch.jit.load(self.model_path, map_location=torch.device('cpu') )
        model.eval()
        model.to(self.device)
        return model

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image

    def infer(self, image):
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image)
            prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
        return prediction

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        prediction = self.infer(image)
        return image_path, prediction


class CustomInfererWrapper:

    @staticmethod
    def run(image_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        customInferer = CustomInferer(model_path="weightFiles/quantized_custom_model.pt", num_classes=NUM_CLASSES, device=device)
        _, prediction = customInferer.predict(image_path)
        return prediction