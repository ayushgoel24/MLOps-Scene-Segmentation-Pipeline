import numpy as np
import torch

NUM_CLASSES = 19
IGNORE_INDEX = 255

class meanIoU:
    """
    Class to find the mean IoU using confusion matrix approach
        CFG (Any): object containing num_classes 
        device (torch.device): compute device
    """    
    def __init__(self):
        self.iouMetric = 0.0
        self.numClasses = NUM_CLASSES
        self.ignoreIndex = IGNORE_INDEX

        # placeholder for confusion matrix on entire dataset
        self.confusion_matrix = np.zeros((self.numClasses, self.numClasses))

    def update(self, y_preds: torch.Tensor, labels: torch.Tensor):
        """ Function finds the IoU for the input batch

        Args:
            y_preds (torch.Tensor): model predictions
            labels (torch.Tensor): groundtruth labels        
        Returns
        """
        predictedLabels = torch.argmax(y_preds, dim=1)
        batchConfusionMatrix = self._fast_hist(labels.numpy().flatten(), predictedLabels.numpy().flatten())
        # add batch metrics to overall metrics
        self.confusion_matrix += batchConfusionMatrix

    
    def _fast_hist(self, label_true, label_pred):
        """ function to calculate confusion matrix on single batch """
        mask = (label_true >= 0) & (label_true < self.numClasses)
        hist = np.bincount(
            self.numClasses * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.numClasses ** 2,
        ).reshape(self.numClasses, self.numClasses)
        return hist


    def compute(self):
        """ Returns overall accuracy, mean accuracy, mean IU, fwavacc """ 
        hist = self.confusion_matrix
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        return mean_iu
    
    def reset(self):
        self.iou_metric = 0.0
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))