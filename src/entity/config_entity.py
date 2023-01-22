import os
import sys
from zipfile import Path, ZipFile
from dataclasses import dataclass
from src.constants import *
from datetime import datetime
from from_root import from_root


TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = SOURCE_DIR_NAME
    artifact_dir: str = os.path.join(ARTIFACTS_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    bucket_name: str = BUCKET_NAME
    zip_file_name: str = ZIP_FILE_NAME
    download_dir: str = os.path.join(from_root(), DATA_DIR_NAME, DOWNLOAD_DIR)
    zip_file_path: str = os.path.join(from_root(), download_dir, ZIP_FILE_NAME)
    unzip_data_dir_path: str = os.path.join(
        from_root(), DATA_DIR_NAME, EXTRACTED_DATA_DIR)


@dataclass
class DataPreprocessingConfig:
    data_preprocessing_artifacts_dir: str = os.path.join(from_root(
    ), training_pipeline_config.artifact_dir, DATA_PREPROCESSING_ARTIFACTS_DIR)
    metadata_dir_path: str = os.path.join(
        data_preprocessing_artifacts_dir, METADATA_DIR)
    metadata_path: str = os.path.join(
        data_preprocessing_artifacts_dir, METADATA_DIR, METADATA_FILE_NAME)
    train_dir_path: str = os.path.join(
        data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TRAIN_DIR)
    train_file_path: str = os.path.join(
        data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TRAIN_DIR, TRAIN_FILE_NAME)
    test_dir_path: str = os.path.join(
        data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TEST_DIR)
    test_file_path: str = os.path.join(
        data_preprocessing_artifacts_dir, DATA_PREPROCESSING_TEST_DIR, TEST_FILE_NAME)
    transformations_dir: str = os.path.join(
        data_preprocessing_artifacts_dir, OTHER_ARTIFACTS)
    transformations_object_path = os.path.join(
        data_preprocessing_artifacts_dir, transformations_dir, TRANSFORMATION_OBJECT_NAME)
    class_mappings_object_path = os.path.join(
        data_preprocessing_artifacts_dir, transformations_dir, CLASS_MAPPINGS_OBJECT_NAME)
    sample_rate: int = SAMPLE_RATE


@dataclass
class CustomDatasetConfig:
    audio_dir: str = os.path.join(
        from_root(), DATA_DIR_NAME, EXTRACTED_DATA_DIR, UNZIPPED_FOLDER_NAME)
    sample_rate: int = SAMPLE_RATE
    num_samples: int = NUM_SAMPLES


@dataclass
class ModelTrainerConfig:
    model_trainer_artifacts_dir: str = os.path.join(
        from_root(), training_pipeline_config.artifact_dir, MODEL_TRAINING_ARTIFACTS_DIR)
    trained_model_dir: str = os.path.join(
        model_trainer_artifacts_dir, TRAINED_MODEL_NAME)
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    stepsize: int = STEP_SIZE
    gamma: float = GAMMA


@dataclass
class ModelEvaluationConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    model_evaluation_artifacts_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR)
    best_model_dir: str = os.path.join(
        model_evaluation_artifacts_dir, S3_MODEL_DIR_NAME)
    in_channels: int = IN_CHANNELS
    base_accuracy: float = BASE_ACCURACY


@dataclass
class ModelPusherConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    model_pusher_artifacts_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR)


@dataclass
class PredictionPipelineConfig:
    s3_model_path: str = S3_BUCKET_MODEL_URI
    prediction_artifact_dir = os.path.join(
        from_root(), PREDICTION_PIPELINE_DIR_NAME)
    model_download_path = os.path.join(
        prediction_artifact_dir, PREDICTION_MODEL_DIR_NAME)
    transformation_download_path = os.path.join(
        prediction_artifact_dir, TRANSFORMATION_ARTIFACTS_DIR)
    app_artifacts = os.path.join(
        prediction_artifact_dir, APPLICATION_ARTIFACTS_DIR)
    input_sounds_path = os.path.join(app_artifacts, 'inputSound.wav')
    wave_sounds_path = os.path.join(app_artifacts, 'input-wave.wav')
