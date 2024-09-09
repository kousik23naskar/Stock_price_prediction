import os
import sys
import pandas as pd
from pathlib import Path
from StockPricePrediction.logger import logging
from StockPricePrediction.exception import AppException
from StockPricePrediction.utils.main_utils import get_size
from StockPricePrediction.entity.config_entity import DataValidationConfig
from StockPricePrediction.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact

class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
        except Exception as e:
            raise AppException(e, sys)

    def validate_data_file(self) -> bool:
        try:
            data_csv_path = self.data_ingestion_artifact.data_csv_file_path
            if not os.path.exists(data_csv_path):
                raise FileNotFoundError(f"File not found: {data_csv_path}")

            file_size_str = get_size(Path(data_csv_path))
            logging.info(f"File size: {file_size_str}")
            
            file_size = os.path.getsize(data_csv_path)

            # Check if the file size is not zero
            file_size_check = file_size != 0

            df = pd.read_csv(data_csv_path)
            required_columns = {'Date', 'Adj Close'}
            columns_present = set(df.columns)
            columns_check = required_columns.issubset(columns_present)
            
            # Validation status is true only if both checks pass
            validation_status = file_size_check and columns_check

            # Save the validation status to a file
            os.makedirs(self.data_validation_config.data_validation_dir, exist_ok=True)
            with open(self.data_validation_config.valid_status_file_dir, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise AppException(e, sys)        

    def initiate_data_validation(self) -> DataValidationArtifact: 
        logging.info("Entered initiate_data_validation method of DataValidation class")
        try:
            status = self.validate_data_file()
            data_validation_artifact = DataValidationArtifact(
                validation_status=status
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise AppException(e, sys)

