import logging
import os

from from_root import from_root
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
os.makedirs(os.path.join(from_root(), "logs"), exist_ok=True)

logs_dir_path = os.path.join(from_root(), "logs")

LOG_FILE_PATH = os.path.join(logs_dir_path, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
