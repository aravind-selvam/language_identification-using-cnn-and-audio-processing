from src.logger import logging
from src.exceptions import CustomException
import torch.nn as nn
import torch.nn.functional as F
import torch

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        """Initialize training and returns loss for each epoch"""
        images, labels = batch 
        # Generate predictions
        out = self(images)
        # Calculate loss                 
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        """
        Takes in Batch size and returns validation loss and accuracy for each batch
        """
        images, labels = batch
        # Generate predictions
        out = self(images)
        # Calculate loss
        loss = F.cross_entropy(out, labels)
        # Calculate accuracy
        acc = accuracy(out, labels)           
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        """Takes a list of outputs and returns mean loss and mean accuracy"""
        batch_losses = [x['val_loss'] for x in outputs]
        # Combine losses and calculate mean loss
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        # Combine accuracies and calculate mean accuracy
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        """Takes the epoch and results to return print results for each epoch"""
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    """Calculates accuracy for the given outputs"""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))