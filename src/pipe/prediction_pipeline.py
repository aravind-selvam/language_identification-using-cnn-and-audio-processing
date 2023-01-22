import os
import sys

import torch
import torchaudio

from src.cloud_storage.s3_operations import S3Sync
from src.constants import *
from src.entity.config_entity import (CustomDatasetConfig,
                                      DataPreprocessingConfig,
                                      PredictionPipelineConfig)
from src.exceptions import CustomException
from src.logger import logging
from src.models.final_model import CNNNetwork
from src.utils import load_object


class LanguageData:
    """
    Custom dataset class for prediction pipeline
    """
    def __init__(self):
        try:
            self.dataset_config = CustomDatasetConfig()
            self.s3_sync = S3Sync()
            self.prediction_config = PredictionPipelineConfig()
        except Exception as e:
            raise CustomException(e,sys)

    def _cut_if_necessary(self, signal):
        try:
            num_samples = self.dataset_config.num_samples
            if signal.shape[1] > num_samples:
                signal = signal[:, :num_samples]
            return signal
        except Exception as e:
            raise CustomException(e,sys)
    
    def _right_pad_if_necessary(self, signal):
        try:
            length_signal = signal.shape[1]
            num_samples = self.dataset_config.num_samples
            if length_signal < num_samples:
                num_missing = num_samples - length_signal
                last_dim_padding = (0, num_missing)
                signal = torch.nn.functional.pad(signal, last_dim_padding)
            return signal
        except Exception as e:
            raise CustomException(e,sys)
        
    def _resample_if_necessary(self, signal, sr):
        try: 
            target_sample_rate = self.dataset_config.sample_rate
            if sr != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                signal = resampler(signal)
            return signal
        except Exception as e:
            raise CustomException(e,sys)
    
    def _mix_down_if_necessary(self, signal):
        try: 
            if signal.shape[0] > 1:
                signal = torch.mean(signal, dim = 0, keepdim=True)
            return signal
        except Exception as e:
            raise CustomException(e,sys)

    def load_data(self, audio_data_path):
        try: 
            transformation_folder = self.prediction_config.transformation_download_path
            os.makedirs(os.path.dirname(transformation_folder), exist_ok=True)
            self.s3_sync.sync_folder_from_s3(folder=transformation_folder, aws_bucket_url=S3_ARTIFACTS_URI)
            transformation_object = os.path.join(transformation_folder, TRANSFORMATION_OBJECT_NAME)
            transformation = load_object(file_path=transformation_object)
            signal, sr = torchaudio.load(audio_data_path)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            signal = self._cut_if_necessary(signal)
            signal = self._right_pad_if_necessary(signal)
            signal = transformation(signal)
            return signal
        except Exception as e:
            raise CustomException(e,sys)

class SinglePrediction:
    """
    Single prediction class for single prediction used in application
    """
    def __init__(self, prediction_pipeline_config: PredictionPipelineConfig):
        try: 
            self.prediction_pipeline_config = prediction_pipeline_config
            self.data_preprocessing_config = DataPreprocessingConfig()
            self.s3_sync = S3Sync()
        except Exception as e:
            raise CustomException(e, sys)

    def _get_model_in_production(self):
        """
        It checks if the model is available in the s3 bucket, if available, it downloads it to the local machine
        and returns the path to the model
        
        Returns:
          The path to the model.
        """
        try:
            s3_model_path = self.prediction_pipeline_config.s3_model_path
            model_download_path = self.prediction_pipeline_config.model_download_path
            os.makedirs(model_download_path, exist_ok=True)
            self.s3_sync.sync_folder_from_s3(folder=model_download_path, aws_bucket_url=s3_model_path)
            for file in os.listdir(model_download_path):
                if file.endswith(".pt"):
                    prediction_model_path = os.path.join(model_download_path, file)
                    logging.info(f"Production model for prediction found in {prediction_model_path}")
                    break
                else:
                    logging.info("Model is not available in Prediction artifacts")
                    prediction_model_path = None
            return prediction_model_path
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def prediction_step(model, input_signal, class_mapping):
        """
        The function takes in a model, an input signal, and a class mapping. It then runs the model on the
        input signal, and returns the language that corresponds to the predicted label.
        
        Args:
          model: the model that you want to use for prediction
          input_signal: the audio signal that we want to classify
          class_mapping: a dictionary of the form {'language': label}
        
        Returns:
          The language that is being predicted.
        """
        try: 
            model.eval()
            with torch.no_grad():
                prediction = model(input_signal)
                prediction_index = prediction[0].argmax(0)
                logging.info("prediction index: {}".format(prediction_index.item()))
                for language, label in class_mapping.items():
                    if label == prediction_index.item():
                        return language
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_model(self):
        """
        It loads the model from the path and returns the model object for prediction.
        
        Returns:
          A model object
        """
        try: 
            prediction_model_path = self._get_model_in_production()
            if prediction_model_path is None:
                return None
            else:
                num_classes = NUM_CLASSES
                in_channels = IN_CHANNELS
                prediction_model = CNNNetwork(in_channels=in_channels, num_classes=num_classes)
                model_state_dict = torch.load(prediction_model_path, map_location='cpu')
                prediction_model.load_state_dict(model_state_dict['model_state_dict'])
                prediction_model.eval()
            return prediction_model
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_language(self, input_signal):
        """
        It downloads the model and the class mappings from S3, and then uses the model to predict the
        language of the input signal
        
        Args:
          input_signal: The input signal is the text that you want to predict the language for.
        
        Returns:
          The output is a dictionary with the following keys:
        """
        try: 
            prediction_model = self.get_model()
            os.makedirs(self.prediction_pipeline_config.prediction_artifact_dir, exist_ok=True)
            download_path = self.prediction_pipeline_config.transformation_download_path
            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            if len(os.listdir(download_path)) > 0:
                self.s3_sync.sync_folder_from_s3(folder=download_path, aws_bucket_url=S3_ARTIFACTS_URI)
                class_mappings_path = os.path.join(download_path,CLASS_MAPPINGS_OBJECT_NAME)
                class_mapping = load_object(file_path=class_mappings_path)
            if prediction_model is not None:
                output = self.prediction_step(prediction_model, input_signal, class_mapping)
                return output
            else:
                raise CustomException("Model not Found in production", sys)
        except Exception as e:
            raise CustomException(e, sys)from e
