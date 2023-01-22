import os
import sys
from typing import Tuple

import numpy as np
import torch

from src.cloud_storage.s3_operations import S3Sync
from src.entity.artifact_entity import (DataPreprocessingArtifacts,
                                        ModelEvaluationArtifacts,
                                        ModelTrainerArtifacts)
from src.entity.config_entity import ModelEvaluationConfig, ModelTrainerConfig
from src.exceptions import CustomException
from src.logger import logging


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts) -> None:
        try:
            self.model_evaluation_config = model_evaluation_config
            self.trainer_artifacts = model_trainer_artifacts
        except CustomException as e:
            raise CustomException(e, sys)

    def get_best_model_path(self) -> str:
        """
        It downloads the best model or production model from S3 and returns the path of the best model

        Returns:
          The best model path is being returned.
        """
        try:
            model_evaluation_artifacts_dir = self.model_evaluation_config.model_evaluation_artifacts_dir
            os.makedirs(model_evaluation_artifacts_dir, exist_ok=True)
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(
                folder=best_model_dir, aws_bucket_url=model_path)
            for file in os.listdir(best_model_dir):
                if file.endswith(".pt"):
                    best_model_path = os.path.join(best_model_dir, file)
                    logging.info(f"Best model found in {best_model_path}")
                    break
                else:
                    logging.info(
                        "Model is not available in best_model_directory")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self) -> Tuple[float, float]:
        """
        It loads the best model from best_model_path and returns s3_accuracy and s3_loss

        Returns:
          The model accuracy and loss
        """
        try:
            best_model_path = self.get_best_model_path()
            if best_model_path is not None:
                state_dict = torch.load(best_model_path, map_location='cpu')
                accuracy = state_dict['accuracy']
                loss = state_dict['loss']
                logging.info(f"S3 Model Validation accuracy is {accuracy}")
                logging.info(f"S3 Model Validation loss is {loss}")
                logging.info(
                    f"Locally trained accuracy is {self.trainer_artifacts.model_accuracy}")
                s3_model_accuracy = accuracy
                s3_model_loss = loss
            else:
                logging.info(
                    "Model is not found on production server, So couldn't evaluate")
                s3_model_accuracy = None
                s3_model_loss = None
            return s3_model_accuracy, s3_model_loss
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Initiates the model evaluation class

        Returns: ModelEvaluationArtifacts dataclass object
        """
        try:
            s3_model_accuracy, s3_model_loss = self.evaluate_model()
            tmp_best_model_accuracy = 0 if s3_model_accuracy is None else s3_model_accuracy
            tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss
            trained_model_accuracy = self.trainer_artifacts.model_accuracy
            trained_model_loss = self.trainer_artifacts.model_loss
            evaluation_response = trained_model_accuracy > tmp_best_model_accuracy and tmp_best_model_loss > trained_model_loss
            model_evaluation_artifacts = ModelEvaluationArtifacts(trained_model_accuracy=trained_model_accuracy,
                                                                  s3_model_accuracy=s3_model_accuracy,
                                                                  is_model_accepted=evaluation_response,
                                                                  trained_model_path=self.trainer_artifacts.trained_model_path,
                                                                  s3_model_path=self.model_evaluation_config.s3_model_path
                                                                  )
            logging.info(
                f"Model evaluation completed! Artifacts: {model_evaluation_artifacts}")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)
