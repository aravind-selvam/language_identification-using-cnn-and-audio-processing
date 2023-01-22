import os
import sys

import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.entity.artifact_entity import ModelTrainerArtifacts
from src.entity.config_entity import ModelTrainerConfig
from src.exceptions import CustomException
from src.logger import logging
from src.utils.gpu_functions import (DeviceDataLoader, get_default_device,
                                     to_device)


# This class is used to train the model using the batch data provided by the data loader class
class ModelTrainer:
    def __init__(self, modeltrainer_config: ModelTrainerConfig,
                optimizer_func: torch.optim.Adam,
                model,
                train_data,
                test_data)-> None:
            try: 
                self.model_trainer_config = modeltrainer_config
                self.learning_rate = modeltrainer_config.learning_rate 
                self.epochs = modeltrainer_config.epochs
                self.optimizer_func = optimizer_func
                self.train_data = train_data
                self.test_data = test_data
                self.device = get_default_device()
                self.model = to_device(model, device=self.device)
            except Exception as e:
                raise CustomException(e, sys)
    
    @torch.no_grad()
    def evaluate(self, model, val_loader):
        """
        It takes a model and a validation loader as input, and returns the validation loss and the
        validation accuracy
        
        Args:
          model: The model to be trained.
          val_loader: the validation data loader
        
        Returns:
          The validation loss and the validation accuracy.
        """
        try:
            model.eval()
            outputs = [model.validation_step(batch) for batch in val_loader]
            return model.validation_epoch_end(outputs)
        except Exception as e:
            raise CustomException(e, sys)

    def fit(self, train_loader, val_loader)-> list:
        """
        It takes a model, a training data loader, and a validation data loader, and trains the model for the
        specified number of epochs
        
        Args:
          train_loader: the training data loader
          val_loader: the validation data loader
        
        Returns:
          The history of the training process.
        """
        try:
            history = []
            self.model.train()
            optimizer = self.optimizer_func(self.model.parameters(), self.learning_rate)
            scheduler = StepLR(optimizer= optimizer, 
                               step_size= self.model_trainer_config.stepsize, 
                               gamma= self.model_trainer_config.gamma
                               )
            for epoch in range(1, self.epochs + 1):
                # Training
                print('Epoch:', epoch,'LR:', scheduler.get_last_lr())
                train_losses = []
                loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
                for batch_idx, batch in loop:
                    loss = self.model.training_step(batch)
                    train_losses.append(loss)
                    
                    # backward 
                    optimizer.zero_grad()
                    loss.backward()

                    # gradient descent
                    optimizer.step()
                    scheduler.step()

                    # update progress bar
                    loop.set_description(f"Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss = loss.item())
                
                # Validation
                result = self.evaluate(self.model, tqdm(val_loader))
                result['train_loss'] = torch.stack(train_losses).mean().item()
                self.model.epoch_end(epoch, result)
                history.append(result)
            return history
        except Exception as e:
            raise CustomException(e, sys)

    def get_dataloader(self) -> DataLoader:
        """
        It returns a tuple of two DataLoader objects, one for training and one for validation
        
        Returns:
          A tuple of two DataLoader objects.
        """
        try:
            train_loader = DataLoader(self.train_data,
                                    batch_size=self.model_trainer_config.batch_size,
                                    shuffle=True,
                                    num_workers=self.model_trainer_config.num_workers,
                                    pin_memory=True)
            
            val_loader = DataLoader(self.test_data,
                                    batch_size=self.model_trainer_config.batch_size * 2,
                                    shuffle=False,
                                    num_workers=self.model_trainer_config.num_workers,
                                    pin_memory=True)
            return train_loader, val_loader
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        """
        The function initiates the model trainer component
        
        Returns:
          The model_trainer_artifacts is being returned.
        """
        try:
            logging.info("Starting model trainer component...")

            train_loader, val_loader = self.get_dataloader()
            # use the wrapper class to load the data to device
            train_dataloader = DeviceDataLoader(train_loader, self.device)
            test_dataloader = DeviceDataLoader(val_loader, self.device)

            # training the model for defined epochs
            history = self.fit(train_dataloader, test_dataloader)
            trained_model_path = self.model_trainer_config.trained_model_dir
            trained_model_dir = self.model_trainer_config.model_trainer_artifacts_dir
            os.makedirs(trained_model_dir, exist_ok=True)
            torch.save({"model_state_dict":self.model.state_dict(),
                        "accuracy":history[0]['val_acc'], 
                        "loss": history[0]['val_loss']}, trained_model_path)
                
            model_trainer_artifacts = ModelTrainerArtifacts(trained_model_path=trained_model_dir,
                                                            model_accuracy=history[0]['val_acc'],
                                                            model_loss=history[0]['val_loss'] )
            logging.info("Model Trainer class completed!!")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)
            