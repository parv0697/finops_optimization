import numpy as np
import pandas as pd
from typing import Dict

def calculate_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates Mean Absolute Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def calculate_mse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates Mean Squared Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean((y_true - y_pred)**2)

def calculate_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculates Root Mean Squared Error."""
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculates Mean Absolute Percentage Error.
    Handles cases where y_true is zero to avoid division by zero.
    Result is in percentage (0-100).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # Avoid division by zero: replace 0s in y_true with a very small number or handle as per specific requirement
    # Here, if y_true is 0, the error for that point is considered 0 if y_pred is also 0, otherwise it's large.
    # A common approach is to ignore points where y_true is 0 or use a variant like MASE.
    # For MAPE, if y_true is 0, that point is problematic.
    # We'll filter out true zeros for MAPE calculation, or return a large value/NaN if all are zero.

    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask): # All true values are zero
        return np.nan # Or 0 if y_pred is also all zero, or a large number otherwise

    y_true_nz = y_true[non_zero_mask]
    y_pred_nz = y_pred[non_zero_mask]

    if len(y_true_nz) == 0: # Should be caught by above, but as a safeguard
        return np.nan

    return np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100

def calculate_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculates Symmetric Mean Absolute Percentage Error.
    Result is in percentage (0-200 or 0-100 depending on formulation, this is 0-100).
    Handles zeros in both actuals and forecasts.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 # Divide by 2 to keep range 0-100 for typical cases.
                                                       # Some formulations omit /2, making range 0-200.

    # Handle cases where denominator is 0 (i.e., y_true and y_pred are both 0)
    # In such cases, the error is 0.
    smape_terms = np.where(denominator == 0, 0, numerator / denominator)
    return np.mean(smape_terms) * 100


def evaluate_all_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calculates all defined performance metrics.
    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
    Returns:
        Dict[str, float]: A dictionary with metric names as keys and their values.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if y_true.empty or y_pred.empty:
        return { # Return NaNs or default if no data to evaluate
            "mae": np.nan,
            "mse": np.nan,
            "rmse": np.nan,
            "mape": np.nan,
            "smape": np.nan
        }

    metrics = {
        "mae": calculate_mae(y_true, y_pred),
        "mse": calculate_mse(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
        "smape": calculate_smape(y_true,y_pred)
    }
    return metrics

if __name__ == '__main__':
    # Example Usage
    true_values = pd.Series([10, 12, 15, 13, 17, 20, 0, 22])
    pred_values = pd.Series([11, 13, 14, 12, 16, 19, 0, 25])
    pred_values_with_zero = pd.Series([11, 13, 14, 12, 16, 19, 5, 25])


    print("--- Basic Test ---")
    metrics = evaluate_all_metrics(true_values, pred_values)
    for metric_name, value in metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    print("\n--- Test with zero in true_values, non-zero in pred_values for that point (for MAPE) ---")
    metrics_zero_true = evaluate_all_metrics(true_values, pred_values_with_zero)
    for metric_name, value in metrics_zero_true.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    print("\n--- Test with all zeros (for MAPE edge case) ---")
    true_zeros = pd.Series([0,0,0])
    pred_some = pd.Series([0,1,0])
    metrics_all_zeros = evaluate_all_metrics(true_zeros, pred_some)
    for metric_name, value in metrics_all_zeros.items():
        print(f"{metric_name.upper()}: {value:.4f}")

    # Test sMAPE
    print("\n--- Test SMAPE ---")
    # Case 1: Perfect match
    print("SMAPE (Perfect Match):", calculate_smape(pd.Series([1,2,3]), pd.Series([1,2,3]))) # Expected: 0
    # Case 2: Some diff
    print("SMAPE (Some Diff):", calculate_smape(pd.Series([100,100]), pd.Series([90,110]))) # Expected: mean( (10 / (95)) , (10 / (105)) ) * 100 approx 10
    # Case 3: Zeroes involved
    print("SMAPE (Actual 0, Pred 0):", calculate_smape(pd.Series([0]), pd.Series([0]))) # Expected: 0
    print("SMAPE (Actual 0, Pred 100):", calculate_smape(pd.Series([0]), pd.Series([100]))) # Expected: (100 / 50) * 100 = 200, then mean. If /2 in formula, then 100. (My formula has /2 -> 100)
    print("SMAPE (Actual 100, Pred 0):", calculate_smape(pd.Series([100]), pd.Series([0]))) # Expected: (100 / 50) * 100 = 200, then mean. If /2 in formula, then 100. (My formula has /2 -> 100)

    # Test empty series
    print("\n--- Test Empty Series ---")
    empty_metrics = evaluate_all_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))
    for metric_name, value in empty_metrics.items():
        print(f"{metric_name.upper()}: {value:.4f}")
