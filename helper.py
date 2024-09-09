import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from StockPricePrediction.pipeline.training_pipeline import TrainPipeline

# Create an instance of the TrainPipeline class
pipeline = TrainPipeline()

def plot_forecast(forecast_plot_path: str) -> BytesIO:
    """
    Read the forecast plot image and return it as a BytesIO object.
    """
    try:
        # Read the image file
        with open(forecast_plot_path, "rb") as file:
            img_bytes = file.read()
        return BytesIO(img_bytes)
    except Exception as e:
        raise Exception(f"Error in reading forecast plot: {e}")
    
def train_stock_model(ticker: str, start_date: str, end_date: str, forecast_periods: int):
    """
    Train the stock model.
    """
    try:
        pipeline.run_pipeline(ticker=ticker, start_date=start_date, end_date=end_date, forecast_periods=forecast_periods)
        return True
    except Exception as e:
        raise Exception(f"Error during training: {e}")
    
def forecast_stock_model(ticker: str, forecast_periods: int):
    """
    Generate forecast and components plots after training.
    """
    try:
        forecast_plot_path = os.path.join(pipeline.model_trainer_config.model_trainer_dir, 'forecast_plot.png')
        components_plot_path = forecast_plot_path.replace('.png', '_components.png')

        if os.path.exists(forecast_plot_path) and os.path.exists(components_plot_path):
            forecast_img = plot_forecast(forecast_plot_path)
            components_img = plot_forecast(components_plot_path)
            return forecast_img, components_img
        else:
            raise FileNotFoundError("Forecast plot or components plot not found.")
    except Exception as e:
        raise Exception(f"Error during forecasting: {e}")