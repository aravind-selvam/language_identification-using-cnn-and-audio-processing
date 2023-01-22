import os
import sys
from zipfile import Path, ZipFile

from src.cloud_storage.s3_operations import S3Sync
from src.constants import S3_BUCKET_URI, UNZIPPED_FOLDER_NAME
from src.entity.artifact_entity import DataIngestionArtifacts
from src.entity.config_entity import DataIngestionConfig
from src.exceptions import CustomException
from src.logger import logging


# > This class is used to ingest data from a source and store it in a database.
class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        The function takes in a DataIngestionConfig object and creates an S3Sync object.

        Args:
          data_ingestion_config (DataIngestionConfig): This is a class that contains all the parameters
        needed to run the job.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = S3Sync()
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_from_cloud(self) -> None:
        """
        It downloads a zip file from an S3 bucket to a local directory

        Returns:
          Stores the data in data ingestion artifact
        """
        try:
            logging.info("Initiating data download from s3 bucket...")
            download_dir = self.data_ingestion_config.download_dir
            zip_file_path = self.data_ingestion_config.zip_file_path
            if os.path.isfile(zip_file_path):
                logging.info(
                    f"Data is already present in {download_dir}, So skipping download step.")
                return None
            else:
                self.s3_sync.sync_folder_from_s3(
                    folder=download_dir, aws_bucket_url=S3_BUCKET_URI)
                logging.info(
                    f"Data is downloaded from s3 bucket to Download directory: {download_dir}.")
        except Exception as e:
            raise CustomException(e, sys)

    def unzip_data(self) -> Path:
        """
        It unzips the downloaded zip file from the download directory and returns the unzipped file path

        Returns:
          The unzip_data_path is being returned.
        """
        try:
            logging.info(
                "Unzipping the downloaded zip file from download directory...")
            zip_file_path = self.data_ingestion_config.zip_file_path
            extract_dir_path = self.data_ingestion_config.unzip_data_dir_path
            unzip_data_path = os.path.join(
                extract_dir_path, UNZIPPED_FOLDER_NAME)
            if os.path.isdir(unzip_data_path):
                logging.info(
                    "Unzipped Folder already exists in unzip directory, so skipping unzip operation.")
            else:
                os.makedirs(extract_dir_path, exist_ok=True)
                with ZipFile(zip_file_path, 'r') as zip_file_ref:
                    zip_file_ref.extractall(extract_dir_path)
            logging.info(
                f"Unzipped file exists in unzip directory: {extract_dir_path}.")
            return unzip_data_path
        except Exception as e:
            raise CustomException(e, sys)

    def rename_files(self) -> None:
        """
        It renames all the files in the unzipped folder to a single format
        """
        try:
            logging.info(
                "Renaming files in unzip directory to single format...")
            extract_dir_path = self.data_ingestion_config.unzip_data_dir_path
            unzip_data_path = os.path.join(
                extract_dir_path, UNZIPPED_FOLDER_NAME)
            for folder in os.listdir(unzip_data_path):
                class_path = unzip_data_path + '/' + str(folder)
                for count, files in enumerate(os.listdir(class_path)):
                    try:
                        dst = f"{folder}-{str(count)}.mp3"
                        src = f"{unzip_data_path}/{folder}/{files}"
                        dst = f"{unzip_data_path}/{folder}/{dst}"
                        os.rename(src, dst)
                    except FileExistsError:
                        pass
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        On calling the class method it initiates the data ingestion class

        Returns:
          DataIngestionArtifacts
        """
        logging.info("Starting data ingestion component...")
        try:
            self.get_data_from_cloud()
            unzip_data_path = self.unzip_data()
            self.rename_files()
            data_ingestion_artifacts = DataIngestionArtifacts(downloaded_data_path=self.data_ingestion_config.download_dir,
                                                              extracted_data_path=unzip_data_path)
            logging.info("Data ingestion completed successfully... \
                        Note: If data is not downloaded try deleting the data folder and try again.")
            return data_ingestion_artifacts
        except Exception as e:
            logging.error(
                "Error in Data Ingestion component! Check above logs")
            raise CustomException(e, sys)
