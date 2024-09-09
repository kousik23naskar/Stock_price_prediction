import sys
from StockPricePrediction.logger import logging
from StockPricePrediction.exception import AppException
from StockPricePrediction.components.data_ingestion import DataIngestion
from StockPricePrediction.components.data_validation import DataValidation
from StockPricePrediction.components.model_trainer import ModelTrainer
from StockPricePrediction.entity.config_entity import DataIngestionConfig, DataValidationConfig, ModelTrainerConfig
from StockPricePrediction.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact, ModelTrainerArtifact

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def start_data_ingestion(self, ticker: str, start_date: str, end_date: str) -> DataIngestionArtifact:
        try: 
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion(ticker, start_date, end_date)
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise AppException(e, sys)
    
    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        logging.info("Entered the start_data_validation method of TrainPipeline class")
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config,
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            return data_validation_artifact
        except Exception as e:
            raise AppException(e, sys) from e

    def start_model_trainer(self, data_csv_path: str, forecast_periods: int) -> ModelTrainerArtifact:
        try:
            model_trainer = ModelTrainer(
                model_trainer_config=self.model_trainer_config,
            )
            model_trainer_artifact = model_trainer.initiate_model_trainer(data_csv_path, forecast_periods)
            return model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)

    def run_pipeline(self, ticker: str, start_date: str, end_date: str, forecast_periods: int) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion(ticker, start_date, end_date)
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)

            if data_validation_artifact.validation_status:
                logging.info("Now we can run Model trainer")
                model_trainer_artifact = self.start_model_trainer(data_ingestion_artifact.data_csv_file_path, forecast_periods)
                logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            else:
                raise Exception("Your data is not in the correct format")
        except Exception as e:
            raise AppException(e, sys)

