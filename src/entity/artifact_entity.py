from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    downloaded_data_path: str 
    extracted_data_path: str

@dataclass
class DataPreprocessingArtifacts:
    train_metadata_path: str
    test_metadata_path: str
    class_mappings: dict
    transformation_object: object
    num_classes: int

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    model_accuracy: float
    model_loss: float

@dataclass
class ModelEvaluationArtifacts:
    trained_model_accuracy: float
    is_model_accepted: bool
    s3_model_accuracy: float
    trained_model_path: str
    s3_model_path: str

@dataclass
class ModelPusherArtifacts:
    response: dict

