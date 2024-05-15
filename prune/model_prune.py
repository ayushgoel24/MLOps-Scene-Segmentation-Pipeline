import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

from prune.class_accuracy import meanClassificationAccuracyMetric
from utils.model_eval import ModelEvaluation

class ModelPruner:
    
    def __init__(self, model, device, test_loader):
        self.model = model
        self.device = device
        self.test_loader = test_loader
        self.prune_percentages = np.linspace(0.05, 0.90, 8).tolist()

    def layer_unstructured_prune(self, model, prune_percentage):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=prune_percentage)
        return model

    def layer_structured_prune(self, model, prune_percentage):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.ln_structured(module, name='weight', amount=prune_percentage, n=1, dim=0)
        return model

    def global_unstructured_prune(self, model, prune_percentage):
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_percentage)
        return model

    def evaluate_pruned_models(self, prune_method):
        accuracies = []
        best_accuracy = 0
        best_model = None
        
        for perc in self.prune_percentages:
            model = self.load_model()
            pruned_model = prune_method(model, perc)
            accuracy = ModelEvaluation.evaluteOnTestData(pruned_model, None, self.device, self.test_loader, meanClassificationAccuracyMetric, "Test accuracy", "Pruned Model")
            accuracies.append((accuracy, pruned_model))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = pruned_model
        
        return accuracies, best_model

    def run_all_pruning(self):
        l1_unstructured_accuracies = self.evaluate_pruned_models(self.layer_unstructured_prune)
        l1_structured_accuracies = self.evaluate_pruned_models(self.layer_structured_prune)
        global_prune_accuracies = self.evaluate_pruned_models(self.global_unstructured_prune)
        
        return {
            "l1_unstructured": l1_unstructured_accuracies,
            "l1_structured": l1_structured_accuracies,
            "global_unstructured": global_prune_accuracies
        }
    
    def prune_model(self):
        accuracies, best_model = self.evaluate_pruned_models(self.layer_unstructured_prune)
        return accuracies, best_model
