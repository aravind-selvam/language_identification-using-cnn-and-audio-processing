import base64
from src.exceptions import CustomException
import sys, os
import dill

def decodesound(string, filename):
    data = base64.b64decode(string)
    with open(filename, 'wb') as f:
        f.write(data)
        f.close()

def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str) -> object:
    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj

    except Exception as e:
        raise CustomException(e, sys) 