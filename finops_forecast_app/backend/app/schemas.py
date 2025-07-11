# Placeholder for Pydantic schemas
# Used for request/response data validation and serialization

from pydantic import BaseModel
from typing import List, Optional, Dict

class FileUploadResponse(BaseModel):
    message: str
    filepath: str

class ForecastParams(BaseModel):
    filepath: str
    date_column: str
    target_column: str
    group_by_columns: Optional[List[str]] = None
    frequency: str # 'D', 'W', 'M'
    models_to_train: List[str] # e.g., ["ARIMA", "Prophet"]
    forecast_horizon: int = 30 # Number of periods to forecast

class Metric(BaseModel):
    mae: float
    mse: float
    rmse: float
    mape: float

class ModelPerformance(BaseModel):
    model_name: str
    metrics: Metric
    forecast_values: List[float] # Or Dict if dates are needed

class GroupForecastResult(BaseModel):
    group_identifier: Dict[str, str] # e.g., {"x_ResourceGroupName": "RG1"}
    model_performances: List[ModelPerformance]
    best_model: Optional[str] = None

class ForecastResponse(BaseModel):
    results_by_group: List[GroupForecastResult]
    overall_summary: Optional[str] = None # e.g., "Prophet performed best on average"

# More schemas will be added as needed

class PreprocessRequest(BaseModel):
    filepath: str
    date_column: str
    target_column: str
    frequency: str # e.g., 'D', 'W', 'M'
    group_by_columns: Optional[List[str]] = None
    missing_value_method: str = 'ffill' # e.g., 'ffill', 'mean', 'zero', 'interpolate'
    outlier_treatment_method: Optional[str] = 'iqr' # e.g., 'iqr', None

class TrainForecastRequest(BaseModel):
    preprocessing_run_id: str # ID obtained from /preprocess_data endpoint
    # original_filepath: str # Could be useful for context or reloading raw if needed, but run_id is key
    target_column: str # Needed to identify the target in the processed parquet files
    models_to_train: List[str] # e.g., ["ARIMA", "Prophet"]
    forecast_horizon: int = 30 # Number of periods to forecast
    evaluation_split_ratio: float = 0.8 # e.g., 0.8 means 80% train, 20% test for evaluation
                                      # Alternatively, can use forecast_horizon as test set size
    model_params: Optional[Dict[str, Dict[str, Any]]] = None # Optional params per model e.g. {"ARIMA": {"seasonal": True, "m":12}}
    # We also need to know which column in the preprocessed parquet file is the target.
    # The preprocessor saves the target column as is.

# Update Metric to include sMAPE
class Metric(BaseModel):
    mae: float
    mse: float
    rmse: float
    mape: float
    smape: float # Added sMAPE

# ModelPerformance can include actual values for the test set for plotting
class ModelPerformance(BaseModel):
    model_name: str
    metrics: Metric
    forecast_values: Dict[str, float] # Dates (as str) to forecast value mapping
    actual_test_values: Optional[Dict[str, float]] = None # Dates (as str) to actual value for test period
    train_summary: Optional[str] = None # Optional summary from model training

# GroupForecastResult can also include the path to the specific group's data
class GroupForecastResult(BaseModel):
    group_identifier: str # Changed from Dict to str for simplicity, matching preprocessor output
    # original_group_identifier_dict: Optional[Dict[str,str]] = None # If the dict form is still useful
    processed_data_path: str # Path to the parquet file for this group
    model_performances: List[ModelPerformance]
    best_model_by_mape: Optional[str] = None # Example: best model by a specific metric
    best_model_by_rmse: Optional[str] = None
    error_message: Optional[str] = None # If processing this group failed

# ForecastResponse remains largely the same
class OverallForecastResponse(BaseModel): # Renamed to avoid conflict if old one is used
    preprocessing_run_id: str
    results_by_group: List[GroupForecastResult]
    overall_summary: Optional[str] = None # e.g., "Prophet performed best on average across X groups"
    global_errors: Optional[List[str]] = None # Errors not specific to a group
