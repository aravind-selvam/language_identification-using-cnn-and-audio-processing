import os
import sys
from typing import Tuple

import pandas as pd
import torchaudio
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from torchaudio.transforms import MelSpectrogram

from src.cloud_storage.s3_operations import S3Sync
from src.constants import FFT_SIZE, HOP_LENGTH, N_MELS, S3_ARTIFACTS_URI
from src.entity.artifact_entity import (DataIngestionArtifacts,
                                        DataPreprocessingArtifacts)
from src.entity.config_entity import DataPreprocessingConfig
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

# The DataPreprocessing class is responsible for creating the metadata files, splitting the data into
# train and test, and creating the transformation object


class DataPreprocessing:
    def __init__(self, data_preprocessing_config: DataPreprocessingConfig,
                 data_ingestion_artifacts: DataIngestionArtifacts):
        """
        The DataPreprocessing __init__ function takes in two parameters, data_preprocessing_config and data_ingestion_artifacts, and
        assigns them to the class variables data_preprocessing_config and data_ingestion_artifacts.

        Args:
          data_preprocessing_config (DataPreprocessingConfig): This is a class that contains all the
        parameters that are required for the preprocessing.
          data_ingestion_artifacts (DataIngestionArtifacts): This is a class that contains the dataframe and
        the path to the dataframe.
        """
        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.data_ingestion_artifacts = data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys)

    def get_meta_data(self) -> Tuple[pd.DataFrame, dict]:
        """
        It takes the extracted data path, creates a dictionary of the filenames and their corresponding
        labels, converts the dictionary to a dataframe, encodes the labels, saves the dataframe csv 
        to the metadata directory

        Returns:
          a tuple of two objects. The first object is a pandas dataframe and the second object is a
        encoded class mapping dictionary.
        """
        try:
            audio_dir = self.data_ingestion_artifacts.extracted_data_path
            metadata = {}
            for label in os.listdir(audio_dir):
                class_path = audio_dir + '/' + str(label)
                audio_clips = os.listdir(class_path)
                for filename in audio_clips:
                    metadata[filename] = label

            metadata = pd.DataFrame.from_dict(
                metadata, orient='index').reset_index().sort_values(by=0)
            metadata.columns = ['filename', 'foldername']
            le = preprocessing.LabelEncoder()
            metadata['labels'] = le.fit_transform(metadata.foldername)

            os.makedirs(os.path.join(
                self.data_preprocessing_config.metadata_dir_path), exist_ok=True)
            metadata_path = self.data_preprocessing_config.metadata_path
            metadata.to_csv(metadata_path, index=False)
            le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            class_mapping_object_path = self.data_preprocessing_config.class_mappings_object_path
            save_object(file_path=class_mapping_object_path,
                        obj=le_name_mapping)

            return metadata, le_name_mapping
        except Exception as e:
            raise CustomException(e, sys)

    def train_test_split(self, metadata: DataFrame) -> None:
        """
        It takes the metadata dataframe as input, splits it into train and test dataframes, and saves them
        as csv files

        Args:
          metadata (DataFrame): DataFrame
        """
        try:
            split = StratifiedShuffleSplit(
                n_splits=5, train_size=0.7, test_size=0.3, random_state=42)
            for train_index, test_index in split.split(metadata, metadata['labels']):
                strat_train_set = metadata.loc[train_index]
                strat_val_set = metadata.loc[test_index]

            # create train and test directory if it doesn't exist
            os.makedirs(os.path.join(
                self.data_preprocessing_config.train_dir_path), exist_ok=True)
            os.makedirs(os.path.join(
                self.data_preprocessing_config.test_dir_path), exist_ok=True)

            # save train and test metadata
            train_file_path = self.data_preprocessing_config.train_file_path
            test_file_path = self.data_preprocessing_config.test_file_path

            strat_train_set.to_csv(train_file_path, index=False)
            strat_val_set.to_csv(test_file_path, index=False)

        except Exception as e:
            raise CustomException(e, sys)

    def audio_transformations(self) -> MelSpectrogram:
        """
        It creates a MelSpectrogram object, saves it to a file, and returns it.

        Returns:
          The MelSpectrogram object is being returned.
        """
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.data_preprocessing_config.sample_rate,
            n_fft=FFT_SIZE,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        transformation_object_path = self.data_preprocessing_config.transformations_object_path
        save_object(file_path=transformation_object_path, obj=mel_spectrogram)

        return mel_spectrogram

    def initiate_data_preprocessing(self) -> DataPreprocessingArtifacts:
        """Initiate the class DataPreprocessing

        Returns:
          The DataPreprocessingArtifacts is being returned.
        """
        try:
            metadata, mappings = self.get_meta_data()
            self.train_test_split(metadata)
            transformation_object = self.audio_transformations()
            data_preprocessing_artifacts = DataPreprocessingArtifacts(train_metadata_path=self.data_preprocessing_config.train_file_path,
                                                                      test_metadata_path=self.data_preprocessing_config.test_file_path,
                                                                      class_mappings=mappings,
                                                                      transformation_object=transformation_object,
                                                                      num_classes=len(
                                                                          mappings)
                                                                      )
            s3_sync = S3Sync()
            other_artifacts_dir = self.data_preprocessing_config.transformations_dir
            logging.info(
                "Sync transformation files to S3 transformation artifacts folder...")
            s3_sync.sync_folder_to_s3(
                folder=other_artifacts_dir, aws_bucket_url=S3_ARTIFACTS_URI)
            logging.info(
                "Finished Syncing files to S3 transformation artifacts folder")

            return data_preprocessing_artifacts
        except Exception as e:
            raise CustomException(e, sys)
