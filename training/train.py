import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class Trainer:
    def __init__(self, model, train_loader, test_loader, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)

    def _check_if_dataset_exists(self):
        pass

    def train(self, epochs, optimizer, criterion, save_path=None):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for data, targets in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                data, targets = data.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.train_loader)}")
            self.validate()
            if save_path:
                torch.save(self.model.state_dict(), save_path)

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy}%')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))



    results = []    
    min_val_loss = np.Inf
    len_train_loader = len(dataloader_train)
    
    # move student model to target device
    student_model.to(device)


    for epoch in range(N_EPOCHS):
        print(f"Starting {epoch + 1} epoch ...")
        
        # Training
        student_model.train()
        train_loss = 0.0
        for i, (inputs, labels, teacher_model_preds) in tqdm(enumerate(dataloader_train), total=len_train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            teacher_model_preds = teacher_model_preds.to(device)

            # Forward Pass on student model
            student_model_preds = student_model(inputs)
            if(isinstance(student_model_preds, OrderedDict) == True):
              student_model_preds = student_model_preds['out']
            
            # Return the KD Loss
            loss = KDLoss(labels, student_model_preds, teacher_model_preds, criterion, temperature, alpha)
            train_loss += loss.item()
                    
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
            
            # Adjust the Learning Rate
            lr_scheduler.step()

        
        # Validate
        student_model.eval()
        validation_loss = 0.0
        iou = meanIoU(device)

        with torch.no_grad():
          for inputs, labels, _ in dataloader_valid:
            inputs = inputs.to(device)
            labels = labels.to(device)  
            y_preds = student_model(inputs)
            if(isinstance(y_preds, OrderedDict) == True):
              y_preds = y_preds['out']            
            
            # calculate loss
            loss = criterion(y_preds, labels)
            validation_loss += loss.item()
                
            # update batch metric information            
            iou.update(y_preds.cpu().detach(), labels.cpu().detach())

        # compute per batch losses
        train_loss = train_loss / len(dataloader_train)
        validation_loss = validation_loss / len(dataloader_valid)

        # compute metric
        val_iou = iou.compute()

        print(f'Epoch: {epoch+1}, Train Loss:{train_loss:6.5f}, Validation Loss:{validation_loss:6.5f}, "Mean IOU:",{val_iou: 4.2f}%')
        
        # store results
        results.append({'epoch': epoch, 'trainLoss': train_loss, 'validationLoss': validation_loss, "Mean IOU": val_iou})
        
        # if validation loss has decreased and user wants to
        if validation_loss <= min_val_loss:
            min_val_loss = validation_loss
            #if saveModel == True:
                #torch.save(student_model.state_dict(), "temp_KD_student.pt")

    results = pd.DataFrame(results)
    plotTrainingResults(results, "temp_KD_student")                
    return results