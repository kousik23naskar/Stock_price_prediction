import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from StockPricePrediction.logger import logging
from StockPricePrediction.exception import AppException
from StockPricePrediction.utils.main_utils import save_plot
from StockPricePrediction.entity.config_entity import ModelTrainerConfig
from StockPricePrediction.entity.artifacts_entity import ModelTrainerArtifact

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig):
        self.model_trainer_config = model_trainer_config

    def train_and_forecast(self, data_csv_path: str, forecast_periods: int) -> tuple:
        logging.info("Entered train_and_forecast method of ModelTrainer class")
        try:
            df = pd.read_csv(data_csv_path)
            df.rename(columns={'Date': 'ds', 'Adj Close': 'y'}, inplace=True)

            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = model.predict(future)

            fig1 = model.plot(forecast)
            ax1 = fig1.gca()  # Get the current axes
            ax1.set_xlabel('Date')  
            ax1.set_ylabel('Price')  
            #ax1.set_title('Stock Price Forecast')
            ax1.legend()

            fig2 = model.plot_components(forecast)

            forecast_plot_path = os.path.join(self.model_trainer_config.model_trainer_dir, 'forecast_plot.png')
            components_plot_path = forecast_plot_path.replace('.png', '_components.png')
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            save_plot(fig1, forecast_plot_path)
            save_plot(fig2, components_plot_path)
            # fig1.savefig(forecast_plot_path)
            # fig2.savefig(components_plot_path)
            # plt.close(fig1)
            # plt.close(fig2)

            logging.info(f"Model training and forecasting complete. Plot saved to {forecast_plot_path} and {components_plot_path}")

            return forecast_plot_path, components_plot_path
        except Exception as e:
            raise AppException(e, sys)

    def initiate_model_trainer(self, data_csv_path: str, forecast_periods: int) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            forecast_plot_path, components_plot_path = self.train_and_forecast(data_csv_path, forecast_periods)
            model_trainer_artifact = ModelTrainerArtifact(
                forecast_plot_path=forecast_plot_path,
                components_plot_path=components_plot_path
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise AppException(e, sys)