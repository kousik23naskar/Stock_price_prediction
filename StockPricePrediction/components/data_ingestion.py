import os
import yfinance as yf
import sys
#from pathlib import Path
from StockPricePrediction.logger import logging
from StockPricePrediction.exception import AppException
from StockPricePrediction.entity.config_entity import DataIngestionConfig
from StockPricePrediction.entity.artifacts_entity import DataIngestionArtifact

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AppException(e, sys)
        
    def download_data(self, ticker: str, start_date: str, end_date: str) -> str:
        """
        Fetch data from yfinance
        """
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            data_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(data_download_dir, exist_ok=True)
            data_file_name = 'data.csv'
            data_csv_path = os.path.join(data_download_dir, data_file_name)
            data.to_csv(data_csv_path)
            logging.info(f"Data downloaded successfully for {ticker} from {start_date} to {end_date} into file {data_csv_path}!")
            return data_csv_path
        except Exception as e:
            raise AppException(e, sys)
    
    def initiate_data_ingestion(self, ticker: str, start_date: str, end_date: str) -> DataIngestionArtifact:
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try: 
            data_csv_path = self.download_data(ticker, start_date, end_date)
            data_ingestion_artifact = DataIngestionArtifact(
                data_csv_file_path=data_csv_path,
                feature_store_path=os.path.dirname(data_csv_path)
            )
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise AppException(e, sys)

