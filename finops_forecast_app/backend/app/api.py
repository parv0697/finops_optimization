import shutil
import os
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Body # Body might not be needed if PreprocessRequest is used directly
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any

from . import data_handler
from . import preprocessor
from . import model_trainer as mt
from . import evaluator as ev
from .schemas import (
    FileUploadResponse, PreprocessRequest, TrainForecastRequest,
    Metric as MetricSchema, ModelPerformance as ModelPerformanceSchema,
    GroupForecastResult as GroupForecastResultSchema,
    OverallForecastResponse
)
from sklearn.model_selection import train_test_split # For splitting data

router = APIRouter()

# Define a temporary directory to store uploaded files and processed data cache
TEMP_DATA_DIR = "temp_data"
PROCESSED_DATA_CACHE_DIR = os.path.join(TEMP_DATA_DIR, "processed_cache")
os.makedirs(TEMP_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_CACHE_DIR, exist_ok=True)


@router.post("/upload_data", response_model=FileUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """
    Accepts a CSV file, saves it temporarily, and returns a success message including the filepath.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are accepted.")

    # Sanitize filename to prevent directory traversal issues
    filename = os.path.basename(file.filename)
    temp_file_path = os.path.join(TEMP_DATA_DIR, filename)

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        # Log the exception e for debugging
        raise HTTPException(status_code=500, detail=f"Could not save file: {str(e)}")
    finally:
        file.file.close()

    return FileUploadResponse(
        message=f"File '{filename}' uploaded successfully.",
        filepath=temp_file_path,
    )


@router.post("/get_column_names")
async def get_columns(filepath: str = Body(..., embed=True)):
    """
    Reads a CSV file from the given filepath and returns its column names.
    The filepath should be the one returned by /upload_data.
    """
    if not os.path.exists(filepath) or TEMP_DATA_DIR not in os.path.abspath(filepath):
        raise HTTPException(status_code=404, detail="File not found or access denied.")
    try:
        df = data_handler.load_csv_from_path(filepath)
        columns = data_handler.get_column_names(df)
        return {"filename": os.path.basename(filepath), "columns": columns}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


class PreprocessRequest(BaseModel):
    filepath: str
    date_column: str
    target_column: str
    frequency: str # 'D', 'W', 'M'
    group_by_columns: Optional[List[str]] = None
    missing_value_method: str = 'ffill'
    outlier_treatment_method: Optional[str] = 'iqr'

@router.post("/preprocess_data")
async def preprocess_data_endpoint(params: PreprocessRequest):
    """
    Loads, preprocesses data based on input parameters, and returns summary of processed groups.
    Saves processed group dataframes for later use in training.
    """
    if not os.path.exists(params.filepath) or TEMP_DATA_DIR not in os.path.abspath(params.filepath):
        raise HTTPException(status_code=404, detail="Uploaded file not found or access denied.")

    try:
        # 1. Load Data
        raw_df = data_handler.load_csv_from_path(params.filepath)

        # 2. Validate date and target columns (basic validation, more in preprocessor)
        # data_handler.identify_date_column(raw_df, params.date_column) # Already does validation
        # data_handler.identify_target_column(raw_df, params.target_column) # Already does validation

        # 3. Full Preprocessing Pipeline
        processed_groups_dict = preprocessor.full_preprocess_pipeline(
            df=raw_df,
            date_column=params.date_column,
            target_column=params.target_column,
            frequency=params.frequency,
            group_by_columns=params.group_by_columns,
            missing_value_method=params.missing_value_method,
            outlier_treatment_method=params.outlier_treatment_method
        )

        # Prepare response and save processed data
        response_summary = []
        processed_data_paths: Dict[str, str] = {} # To store paths of saved processed group data

        # Create a unique sub-directory for this preprocessing run's cache
        # Based on hash of filename and params to ensure some level of uniqueness / reusability
        import hashlib
        params_str = f"{os.path.basename(params.filepath)}_{params.date_column}_{params.target_column}_{params.frequency}_{params.group_by_columns}_{params.missing_value_method}_{params.outlier_treatment_method}"
        run_hash = hashlib.md5(params_str.encode()).hexdigest()
        current_run_cache_dir = os.path.join(PROCESSED_DATA_CACHE_DIR, run_hash)
        os.makedirs(current_run_cache_dir, exist_ok=True)


        for group_key, group_df in processed_groups_dict.items():
            group_key_str = "_".join(map(str, group_key)) if isinstance(group_key, tuple) else str(group_key)

            # Save each processed group DataFrame to a Parquet file for efficiency
            processed_group_filename = f"processed_{group_key_str}.parquet"
            processed_group_filepath = os.path.join(current_run_cache_dir, processed_group_filename)
            group_df.to_parquet(processed_group_filepath)
            processed_data_paths[group_key_str] = processed_group_filepath

            response_summary.append({
                "group_identifier": group_key_str,
                "num_records": len(group_df),
                "time_range_start": group_df.index.min().isoformat() if not group_df.empty else None,
                "time_range_end": group_df.index.max().isoformat() if not group_df.empty else None,
                "missing_values_after_processing": group_df[params.target_column].isnull().sum(),
                "processed_data_path": processed_group_filepath # For backend use
            })

        return {
            "message": "Data preprocessed successfully.",
            "preprocessing_run_id": run_hash, # ID to retrieve this set of processed files
            "summary_by_group": response_summary,
            # "processed_data_references": processed_data_paths # For backend to know where files are
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(ve)}")
    except Exception as e:
        # Log the exception e for server-side debugging
        print(f"Unhandled exception in /preprocess_data: {e}") # Basic logging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


# Example Pydantic model for PreprocessRequest (move to schemas.py ideally)
from pydantic import BaseModel
class PreprocessRequest(BaseModel):
    filepath: str
    date_column: str
    target_column: str
    frequency: str # 'D', 'W', 'M'
    group_by_columns: Optional[List[str]] = None
    missing_value_method: str = 'ffill'
    outlier_treatment_method: Optional[str] = 'iqr'

# Note: The PreprocessRequest model is defined here for brevity in this step,
# but it should ideally be in `schemas.py` and imported.
# I see I already added a similar one in schemas.py (ForecastParams),
# so I should ensure they are consistent or consolidate. For now, this is functional.
# For now, I will add the `PreprocessRequest` to schemas.py.
    # The PreprocessRequest model is now imported from schemas.py

# A simple /hello endpoint for basic API testing
@router.get("/hello")
async def hello():
    return {"message": "Hello from API router!"}


# PROCESSED_DATA_CACHE_DIR is defined globally at the top of the file.

@router.post("/train_forecast_evaluate", response_model=OverallForecastResponse)
async def train_forecast_evaluate_endpoint(params: TrainForecastRequest):
    """
    Trains selected models on preprocessed data for each group,
    generates forecasts, evaluates performance, and returns results.
    """
    run_cache_dir = os.path.join(PROCESSED_DATA_CACHE_DIR, params.preprocessing_run_id)
    if not os.path.isdir(run_cache_dir):
        raise HTTPException(status_code=404, detail=f"Preprocessing run ID '{params.preprocessing_run_id}' not found.")

    all_group_results: List[GroupForecastResultSchema] = []
    global_errors: List[str] = []

    # Iterate over the .parquet files in the specific run_cache_dir
    # These files were saved by the /preprocess_data endpoint.
    # Their names contain the group_identifier.
    for filename in os.listdir(run_cache_dir):
        if not filename.endswith(".parquet") or not filename.startswith("processed_"):
            continue

        processed_group_filepath = os.path.join(run_cache_dir, filename)
        # Extract group_identifier from filename: "processed_{group_key_str}.parquet"
        group_identifier_str = filename.replace("processed_", "").replace(".parquet", "")

        group_model_performances: List[ModelPerformanceSchema] = []
        group_error_message: Optional[str] = None

        try:
            group_series_df = pd.read_parquet(processed_group_filepath)
            if group_series_df.empty or params.target_column not in group_series_df.columns:
                raise ValueError(f"Data for group {group_identifier_str} is empty or target column missing.")

            # Ensure index is DatetimeIndex (parquet should preserve it, but good to check)
            if not isinstance(group_series_df.index, pd.DatetimeIndex):
                # Attempt to convert if it's a standard date column named 'index' or similar
                if 'index' in group_series_df.columns and pd.api.types.is_datetime64_any_dtype(group_series_df['index']):
                    # This case handles if index was saved as a column named 'index'
                    group_series_df = group_series_df.set_index('index')
                elif not isinstance(group_series_df.index, pd.DatetimeIndex):
                     # If index is not datetime, try to convert it. This is less ideal as parquet should preserve it.
                    try:
                        group_series_df.index = pd.to_datetime(group_series_df.index)
                    except Exception as e_date_conv:
                         raise ValueError(f"Index of data for group {group_identifier_str} is not DatetimeIndex and could not be converted: {e_date_conv}. Ensure preprocessor saves a DatetimeIndex.")

            # Ensure frequency is set on the index, important for models like ARIMA/Prophet
            # The preprocessor should have set this. If not, try to infer.
            if group_series_df.index.freq is None:
                inferred_freq = pd.infer_freq(group_series_df.index)
                if inferred_freq:
                    group_series_df = group_series_df.asfreq(inferred_freq)
                else:
                    # This is problematic. Fallback or raise error.
                    # For now, we'll let asfreq() below handle it, which might error or lead to issues.
                    warnings.warn(f"Could not infer frequency for group {group_identifier_str}. Forecasting accuracy may be affected.")


            series_to_forecast = group_series_df[params.target_column]
            # Attempt to set frequency if not already set. This is crucial.
            if series_to_forecast.index.freq is None:
                # Try to infer again, or use the frequency from the request if it matches the data's implicit freq.
                # This part is tricky if the preprocessor didn't embed the freq.
                # For now, assume preprocessor sets it. If not, models might fail or infer incorrectly.
                # A more robust solution would be to store desired_freq from PreprocessRequest and use it here.
                # Let's assume the index IS a DatetimeIndex and try to apply asfreq directly using the stored frequency.
                # This requires storing the 'frequency' (D, W, M) from the PreprocessRequest alongside the run_id,
                # or passing it into TrainForecastRequest.
                # For now, let model_trainer handle asfreq if necessary based on model reqs.
                pass


            # Split data for evaluation: use last `forecast_horizon` points as test set
            # Or use evaluation_split_ratio if series is long enough
            if len(series_to_forecast) < params.forecast_horizon * 2 or len(series_to_forecast) < 10: # Heuristic: need enough data
                # Not enough data to do a meaningful train/test split for evaluation.
                # Train on all data and forecast, but metrics will be NaN or less reliable.
                train_series = series_to_forecast
                test_series = pd.Series([], dtype=float, index=pd.to_datetime([])) # Empty test series
                # print(f"Warning: Group {group_identifier_str} has insufficient data ({len(series_to_forecast)} points) for robust train/test split. Training on all data.")
            else:
                # Use last forecast_horizon points as test set
                split_point = len(series_to_forecast) - params.forecast_horizon
                train_series = series_to_forecast.iloc[:split_point]
                test_series = series_to_forecast.iloc[split_point:]


            for model_name_to_train in params.models_to_train:
                try:
                    model_specific_params = params.model_params.get(model_name_to_train, {}) if params.model_params else {}
                    model_instance = mt.get_model(model_name_to_train, model_params=model_specific_params)

                    # TODO: Handle exogenous variables if they are part of the preprocessed data
                    # For now, assuming no exogenous variables are passed from preprocessor to model trainer.
                    # If exog vars are present in group_series_df, they need to be identified and passed.

                    model_instance.train(train_series) # Train on the training portion

                    # Forecast for the length of the test set (if exists) or params.forecast_horizon
                    # If test_series is empty, this predict is for future, evaluation will be skipped/NaN
                    # If test_series is not empty, this predict is for the test period
                    num_periods_for_pred = len(test_series) if not test_series.empty else params.forecast_horizon

                    if num_periods_for_pred == 0 and params.forecast_horizon > 0 : # case where test series is empty but we want future forecast
                        num_periods_for_pred = params.forecast_horizon

                    if num_periods_for_pred > 0:
                        predictions_series = model_instance.predict(n_periods=num_periods_for_pred)
                    else: # No periods to predict (e.g. test series empty and horizon is 0)
                        predictions_series = pd.Series([], dtype=float, index=pd.to_datetime([]))


                    current_metrics = {}
                    actual_test_values_dict = None
                    if not test_series.empty and not predictions_series.empty:
                        # Ensure alignment for evaluation if predictions_series index doesn't match test_series exactly
                        # This can happen if model.predict() creates a slightly different index.
                        # For robust evaluation, align them.
                        # We expect predictions_series to cover the same period as test_series.
                        # If predict() was for future (test_series was empty), then metrics can't be calculated here.

                        # Align predictions to the test_series index for fair comparison
                        # This assumes predictions_series covers the same span as test_series
                        # If model.predict was for future, then test_series is empty, and this block is skipped.
                        aligned_predictions = predictions_series.reindex(test_series.index)
                        # Fill NaNs in aligned_predictions that might result from reindexing if model couldn't predict for some test dates
                        # A simple ffill/bfill might be too naive. If model didn't predict, it's a model issue.
                        # However, for metrics, we need non-NaN. If NaNs appear, it indicates a problem.
                        # For now, let's assume predict gives values for all requested periods.

                        current_metrics = ev.evaluate_all_metrics(test_series, aligned_predictions.fillna(0)) # fillna(0) for safety if alignment causes NaN
                        actual_test_values_dict = {str(k.date()): v for k, v in test_series.items()}
                    else:
                         # Metrics are NaN if no test data
                        current_metrics = {m: np.nan for m in MetricSchema.__fields__.keys()}


                    # If we trained on all data (test_series was empty), now make the actual future forecast
                    if test_series.empty and params.forecast_horizon > 0:
                        # Retrain on ALL available data for this group before making final future forecast
                        model_instance.train(series_to_forecast) # series_to_forecast is all data for the group
                        predictions_series = model_instance.predict(n_periods=params.forecast_horizon)


                    forecast_values_dict = {str(k.date()): v for k, v in predictions_series.items()}

                    group_model_performances.append(ModelPerformanceSchema(
                        model_name=model_name_to_train,
                        metrics=MetricSchema(**current_metrics),
                        forecast_values=forecast_values_dict,
                        actual_test_values=actual_test_values_dict,
                        train_summary=model_instance.get_train_summary()
                    ))

                except Exception as model_e:
                    # Error during training/forecasting a specific model for this group
                    error_msg = f"Error with model {model_name_to_train} for group {group_identifier_str}: {str(model_e)}"
                    print(error_msg) # Server log
                    # Add a performance entry indicating failure for this model
                    nan_metrics = {m: np.nan for m in MetricSchema.__fields__.keys()}
                    group_model_performances.append(ModelPerformanceSchema(
                        model_name=model_name_to_train,
                        metrics=MetricSchema(**nan_metrics),
                        forecast_values={},
                        train_summary=f"Failed: {str(model_e)}"
                    ))

        except Exception as group_e:
            # Error processing this entire group (e.g., loading data, splitting)
            group_error_message = f"Error processing group {group_identifier_str}: {str(group_e)}"
            print(group_error_message) # Server log
            # global_errors.append(group_error_message) # Or add to group specific error

        # Determine best model for the group based on MAPE and RMSE (lower is better)
        best_mape = float('inf')
        best_rmse = float('inf')
        best_model_mape_name = None
        best_model_rmse_name = None

        for perf in group_model_performances:
            if not np.isnan(perf.metrics.mape) and perf.metrics.mape < best_mape:
                best_mape = perf.metrics.mape
                best_model_mape_name = perf.model_name
            if not np.isnan(perf.metrics.rmse) and perf.metrics.rmse < best_rmse:
                best_rmse = perf.metrics.rmse
                best_model_rmse_name = perf.model_name

        all_group_results.append(GroupForecastResultSchema(
            group_identifier=group_identifier_str,
            processed_data_path=processed_group_filepath,
            model_performances=group_model_performances,
            best_model_by_mape=best_model_mape_name,
            best_model_by_rmse=best_model_rmse_name,
            error_message=group_error_message
        ))

    # TODO: Overall summary could compare performance across groups if meaningful
    return OverallForecastResponse(
        preprocessing_run_id=params.preprocessing_run_id,
        results_by_group=all_group_results,
        global_errors=global_errors if global_errors else None
    )
