import os.path
import sys
import matplotlib.pyplot as plt
#import yaml
from pathlib import Path
from StockPricePrediction.exception import AppException
from StockPricePrediction.logger import logging


def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def save_plot(fig, file_path: str) -> None:
    try:
        fig.savefig(file_path)
        plt.close(fig)
        
    except Exception as e:
         raise AppException(e, sys) from e

# def read_yaml_file(file_path: str) -> dict:
#     try:
#         with open(file_path, "rb") as yaml_file:
#             logging.info("Read yaml file successfully")
#             return yaml.safe_load(yaml_file)

#     except Exception as e:
#         raise AppException(e, sys) from e


# def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
#     try:
#         if replace:
#             if os.path.exists(file_path):
#                 os.remove(file_path)

#         os.makedirs(os.path.dirname(file_path), exist_ok=True)

#         with open(file_path, "w") as file:
#             yaml.dump(content, file)
#             logging.info("Successfully write_yaml_file")

#     except Exception as e:
#         raise AppException(e, sys) 