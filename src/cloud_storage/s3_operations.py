import os


class S3Sync:

    def sync_folder_to_s3(self, folder, aws_bucket_url):
        """
        It takes a folder and an AWS bucket URL as input, and then it syncs the local folder to the AWS bucket

        Args:
          folder: The folder you want to sync to S3
          aws_bucket_url: The URL of the S3 bucket you want to sync to.
        """
        command = f"aws s3 sync {folder} {aws_bucket_url} "
        os.system(command)

    def sync_folder_from_s3(self, folder, aws_bucket_url):
        """
        It takes a folder and an s3 bucket url as input downloads the file from s3 bucket to local folder

        Args:
          folder: The folder on which you want to download files from S3
          aws_bucket_url: The URL of the S3 bucket you want to download from.
        """
        command = f"aws s3 sync  {aws_bucket_url} {folder} "
        os.system(command)
