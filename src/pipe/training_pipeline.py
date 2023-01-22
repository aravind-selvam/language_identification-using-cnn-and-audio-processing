import torch

from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.dataset_custom import IndianLanguageDataset
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import *
from src.entity.config_entity import *
from src.entity.config_entity import (DataIngestionConfig,
                                      DataPreprocessingConfig)
from src.exceptions import CustomException
from src.logger import logging
from src.models.final_model import CNNNetwork


# The class is used to run the training pipeline
class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()
        self.dataset_config = CustomDatasetConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Starting data ingestion in training pipeline")
        try:
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info(
                "Data ingestion step completed successfully in train pipeline")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_preprocessing(self, data_ingestion_artifacts: DataIngestionArtifacts) -> DataPreprocessingArtifacts:
        logging.info("Starting data preprocessing in training pipeline")
        try:
            data_preprocessing = DataPreprocessing(data_preprocessing_config=self.data_preprocessing_config,
                                                   data_ingestion_artifacts=data_ingestion_artifacts)
            data_preprocessing_artifacts = data_preprocessing.initiate_data_preprocessing()
            logging.info(
                "Data preprocessing step completed successfully in train pipeline")
            return data_preprocessing_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, data_preprocessing_artifacts: DataPreprocessingArtifacts) -> ModelTrainerArtifacts:
        logging.info("Starting model training in training pipeline")
        try:
            logging.info(
                "Instantiating train and validation dataset from custom dataset class...")
            train_data = IndianLanguageDataset(dataset_config=self.dataset_config,
                                               transformations=data_preprocessing_artifacts.transformation_object,
                                               validation=False,
                                               preprocessing_artifacts=data_preprocessing_artifacts)

            test_data = IndianLanguageDataset(dataset_config=self.dataset_config,
                                              transformations=data_preprocessing_artifacts.transformation_object,
                                              validation=True,
                                              preprocessing_artifacts=data_preprocessing_artifacts)

            logging.info("Instantiating CNNNetwork model...")
            model = CNNNetwork(
                in_channels=1, num_classes=data_preprocessing_artifacts.num_classes)

            logging.info("Instantiating model trainer class...")
            model_trainer = ModelTrainer(modeltrainer_config=self.model_trainer_config,
                                         data_preprocessing_artifacts=data_preprocessing_artifacts,
                                         model=model,
                                         train_data=train_data,
                                         test_data=test_data,
                                         optimizer_func=torch.optim.Adam)

            logging.info(
                f"The training pipeline is current running in device: {model_trainer.device}.")
            model_trainer_artifacts = model_trainer.initiate_model_trainer()
            logging.info(
                "Model trainer step completed successfully in train pipeline")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self, model_trainer_artifacts: ModelTrainerArtifacts) -> ModelEvaluationArtifacts:
        logging.info("Starting model evaluation in training pipeline")
        try:
            model_evaluation = ModelEvaluation(model_evaluation_config=self.model_evaluation_config,
                                               model_trainer_artifacts=model_trainer_artifacts)
            logging.info("Evaluating current trained model")
            model_evaluation_artifacts = model_evaluation.initiate_evaluation()
            logging.info(
                "Model evaluation step completed successfully in train pipeline")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        logging.info("Starting model pusher in training pipeline")
        try:
            model_pusher = ModelPusher(
                model_evaluation_artifacts=model_evaluation_artifacts)
            logging.info(
                "If model is accepted in model evaluation. Pushing the model into production storage")
            model_pusher_artifacts = model_pusher.initiate_model_pusher()
            logging.info(
                "Model pusher step completed successfully in train pipeline")
            return model_pusher_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self) -> None:
        """
        The function runs the data ingestion, data preprocessing, model training, model evaluation, and
        model pusher steps in the pipeline and completes the training pipeline
        """
        logging.info(">>>> Initializing training pipeline <<<<")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_preprocessing_artifacts = self.start_data_preprocessing(
                data_ingestion_artifacts=data_ingestion_artifacts)

            model_trainer_artifacts = self.start_model_training(
                data_preprocessing_artifacts=data_preprocessing_artifacts)

            model_evaluation_artifacts = self.start_model_evaluation(model_trainer_artifacts=model_trainer_artifacts,
                                                                     data_preprocessing_artifacts=data_preprocessing_artifacts)

            model_pusher_artifact = self.start_model_pusher(
                model_evaluation_artifacts=model_evaluation_artifacts)

            logging.info("<<<< Training pipeline completed >>>>")
        except Exception as e:
            raise CustomException(e, sys)
