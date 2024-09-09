from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    data_csv_file_path: str
    feature_store_path: str

@dataclass
class DataValidationArtifact:
    validation_status: bool

@dataclass
class ModelTrainerArtifact:
    forecast_plot_path: str
    components_plot_path: str

