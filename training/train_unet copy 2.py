from cityscapesscripts.helpers.labels import trainId2label
import mlflow
import mlflow.pytorch
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from segmentation_models_pytorch import Unet

from logger.manager import LoggerManager
from dataset.cityscapes_dataloader import CityScapes_DataLoader
from mean_iou import meanIoU

NUM_CLASSES = 19

class UnetTrainer:
    
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)
        self.logger = LoggerManager(log_file_name="unet_trainer.log", logger_name="unet_trainer").get_logger()

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

    def train(self, num_epochs, optimizer, criterion, save_path=None):
        with mlflow.start_run():
            mlflow.log_param("epochs", num_epochs)
            mlflow.log_param("batch_size", 4)
            mlflow.log_param("learning_rate", 0.001)

            for epoch in range(num_epochs):
                self.logger.log(f"Running Epoch: {epoch + 1}/{num_epochs}")
                self.model.train()
                running_loss = 0.0
                with tqdm(total=len(self.train_loader), desc=f"Running Epoch: {epoch+1}/{num_epochs}") as pbar: 
                    for images, masks in self.train_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        optimizer.zero_grad()
                        outputs = self.model(images)
                        loss = criterion(outputs, masks)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        pbar.update(1)
                        pbar.set_postfix(loss=loss.item())

                avg_loss = running_loss / len(self.train_loader)
                mlflow.log_metric("loss", avg_loss, step=epoch)
                validation_loss, validation_metric = self.validate(criterion)
                self.logger.log(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

            self.logger.log("Finished Training")
        
        # Log the trained model
        mlflow.pytorch.log_model(self.model, "model")

        # Save the model locally
        torch.save(self.model.state_dict(), "unet_cityscapes.pth")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

class UnetTrainWrapper:

    @staticmethod
    def run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=len(trainId2label)).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        mlflow.set_experiment("Cityscapes_Semantic_Segmentation")

        train_loader, val_loader, test_loader = CityScapes_DataLoader()

        unetTrainer = UnetTrainer(model=model, train_loader=train_loader, test_loader=val_loader, device=device)
        
        unetTrainer.train(num_epochs=10, optimizer=optimizer, criterion=criterion)