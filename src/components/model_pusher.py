import sys

from src.cloud_storage.s3_operations import S3Sync
from src.constants import TRAINED_MODEL_NAME
from src.entity.artifact_entity import (ModelEvaluationArtifacts,
                                        ModelPusherArtifacts)
from src.exceptions import CustomException
from src.logger import logging


# This class is responsible for pushing the trained model to the production server storage if the trained model is accepted
class ModelPusher:
    def __init__(self, model_evaluation_artifacts: ModelEvaluationArtifacts):
        self.model_evaluation_artifacts = model_evaluation_artifacts

    def initiate_model_pusher(self):
        """
        The function initiates the model pusher component and pushes the current trained model to the
        production server storage if the model is accepted by the model evaluation component
        
        Returns:
          The ModelPusherArtifacts class is being returned.
        """
        try:
            logging.info("Initiating model pusher component")
            if self.model_evaluation_artifacts.is_model_accepted:
                trained_model_path = self.model_evaluation_artifacts.trained_model_path
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                s3_sync = S3Sync()
                s3_sync.sync_folder_to_s3(folder=trained_model_path, aws_bucket_url=s3_model_folder_path)
                message = "Model Pusher pushed the current Trained model to Production server storage"
                response = {"is model pushed": True, "S3_model": s3_model_folder_path + "/" + str(TRAINED_MODEL_NAME),"message" : message}
                logging.info(response)
            else:
                s3_model_folder_path = self.model_evaluation_artifacts.s3_model_path
                message = "Current Trained Model is not accepted as model in Production has better accuracy"
                response = {"is model pushed": False, "S3_model":s3_model_folder_path,"message" : message}
                logging.info(response)
            
            model_pusher_artifacts = ModelPusherArtifacts(response=response)
            return model_pusher_artifacts

        except Exception as e:
            raise CustomException(e, sys)
        