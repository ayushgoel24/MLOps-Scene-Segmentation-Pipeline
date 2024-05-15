import torch
import torch.nn as nn

import os
from tqdm import tqdm
from collections import namedtuple, OrderedDict

class ModelEvaluation:

    @staticmethod
    def evaluteOnTestData(model, pretrainedModelPath, device, dataloader_test, metricClass, metricName, modelName, logger, verbose=False):
        """Evaluate the model on test set

        Args:
            model (nn.Module): input model
            pretrainedModelPath (str): path of weight file
            device (torch.device): compute device such as GPU or CPU
            dataloader_test (DataLoader): test dataset
            metricClass : function / class that calculates metric b/w predicted and ground truth  
            metricName (str) : name of metric
            modelName (str): name of the model
            verbose (bool, optional): flag to print results. Defaults to False.

        Returns:
            testSetMetric(float): metric on test data
        """
        testSetMetric = 0.0

        if verbose == True:
            logger.log("------------------------")
            logger.log(f"Test Data Results for {modelName} using {str(device)}")
            logger.log("------------------------")
        
        modelLoadStatus = False
        if pretrainedModelPath is not None:
            if os.path.isfile(pretrainedModelPath) == True:
                model.load_state_dict(torch.load(pretrainedModelPath, map_location=device))
                modelLoadStatus = True
        else:
            modelLoadStatus = True

        if modelLoadStatus == True:
            lenTestLoader = len(dataloader_test)
            model.to(device)
            # set to inference mode
            model.eval()
            metricObject = metricClass(device=device)

            with torch.no_grad():
                for inputs, labels in tqdm(dataloader_test, total=lenTestLoader):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    y_preds = model(inputs)
                
                    if(isinstance(y_preds, OrderedDict) == True):
                        y_preds = y_preds['out']

                    # update batch metric information            
                    metricObject.update(y_preds.cpu().detach(), labels.cpu().detach())

            # compute metric of test set predictions
            testSetMetric = metricObject.compute()
            
            if verbose == True:
                logger.log(f'{modelName} has {testSetMetric} {metricName} on testData')
        else:
            logger.log(f'Model cannot load state_dict from {pretrainedModelPath}')
            
        return testSetMetric