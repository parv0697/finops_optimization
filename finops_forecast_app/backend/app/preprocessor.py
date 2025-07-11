import pandas as pd
from typing import List, Optional, Dict, Union, Tuple

def group_and_resample_time_series(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    frequency: str, # 'D', 'W', 'M'
    group_by_columns: Optional[List[str]] = None,
    aggregation_method: str = 'sum' # e.g., 'sum', 'mean' for target
) -> Dict[Union[str, Tuple[str,...]], pd.DataFrame]:
    """
    Groups data by specified columns and then resamples the time series for each group.

    Args:
        df: Input DataFrame.
        date_column: Name of the column containing date information.
        target_column: Name of the column to be forecasted (numeric).
        frequency: Resampling frequency ('D' for daily, 'W' for weekly, 'M' for monthly).
        group_by_columns: List of column names to group by. If None, processes the whole DataFrame as one series.
        aggregation_method: How to aggregate the target column during resampling (e.g., 'sum', 'mean').

    Returns:
        A dictionary where keys are group identifiers (or a default key if no grouping)
        and values are DataFrames with a DatetimeIndex and the resampled target column.
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        # Attempt conversion, raise error if it fails
        try:
            df[target_column] = pd.to_numeric(df[target_column])
        except ValueError:
            raise ValueError(f"Target column '{target_column}' must be numeric.")

    # Ensure date column is datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        raise ValueError(f"Could not parse date column '{date_column}': {e}")

    grouped_series_dict: Dict[Union[str, Tuple[str,...]], pd.DataFrame] = {}

    if group_by_columns and len(group_by_columns) > 0:
        if not all(col in df.columns for col in group_by_columns):
            missing_cols = [col for col in group_by_columns if col not in df.columns]
            raise ValueError(f"Grouping columns not found in DataFrame: {missing_cols}")

        grouped = df.groupby(group_by_columns)
        for name, group_df in grouped:
            group_key_str = name if isinstance(name, str) else "_".join(map(str, name))
            # print(f"Processing group: {group_key_str}")

            # Set date column as index for resampling
            group_df_indexed = group_df.set_index(date_column)

            # Select only the target column for resampling
            resampled_series = group_df_indexed[[target_column]].resample(frequency).agg(aggregation_method)

            # Rename the target column post-aggregation if needed, or ensure it's consistently named
            # resampled_series = resampled_series.rename(columns={target_column: target_column}) # Already correct

            grouped_series_dict[name] = resampled_series # Use original tuple name as key
    else:
        # No grouping, process the entire DataFrame
        df_indexed = df.set_index(date_column)
        resampled_series = df_indexed[[target_column]].resample(frequency).agg(aggregation_method)
        grouped_series_dict["overall"] = resampled_series # Default key for non-grouped data

    return grouped_series_dict


def handle_missing_values(
    series_df: pd.DataFrame, # DataFrame with DatetimeIndex and one target column
    target_column: str,
    method: str = 'ffill', # 'ffill', 'bfill', 'mean', 'median', 'zero'
    fill_value: Optional[float] = None # Used if method is 'constant'
) -> pd.DataFrame:
    """
    Handles missing values in a time series DataFrame.
    Assumes series_df has a DatetimeIndex and one target column.
    """
    if target_column not in series_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in series DataFrame.")

    series_filled = series_df.copy()

    if method == 'ffill':
        series_filled[target_column] = series_filled[target_column].fillna(method='ffill')
    elif method == 'bfill':
        series_filled[target_column] = series_filled[target_column].fillna(method='bfill')
    elif method == 'mean':
        series_filled[target_column] = series_filled[target_column].fillna(series_df[target_column].mean())
    elif method == 'median':
        series_filled[target_column] = series_filled[target_column].fillna(series_df[target_column].median())
    elif method == 'zero':
        series_filled[target_column] = series_filled[target_column].fillna(0)
    elif method == 'constant' and fill_value is not None:
         series_filled[target_column] = series_filled[target_column].fillna(fill_value)
    elif method == 'interpolate': # Linear interpolation
        series_filled[target_column] = series_filled[target_column].interpolate(method='linear')
    else:
        # Optionally, if there are still NaNs (e.g., ffill at the beginning), fill with 0 or median
        series_filled[target_column] = series_filled[target_column].fillna(0) # Default fallback

    return series_filled


def detect_and_treat_outliers_iqr(
    series_df: pd.DataFrame, # DataFrame with DatetimeIndex and one target column
    target_column: str,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detects outliers using the IQR method and caps them.
    Assumes series_df has a DatetimeIndex and one target column.
    """
    if target_column not in series_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in series DataFrame.")

    series_treated = series_df.copy()
    Q1 = series_treated[target_column].quantile(0.25)
    Q3 = series_treated[target_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Cap outliers
    series_treated[target_column] = series_treated[target_column].clip(lower=lower_bound, upper=upper_bound)
    return series_treated


# Main preprocessing pipeline function
def full_preprocess_pipeline(
    df: pd.DataFrame,
    date_column: str,
    target_column: str,
    frequency: str,
    group_by_columns: Optional[List[str]] = None,
    missing_value_method: str = 'ffill',
    outlier_treatment_method: Optional[str] = 'iqr' # 'iqr' or None
) -> Dict[Union[str, Tuple[str,...]], pd.DataFrame]:
    """
    Applies the full preprocessing pipeline: grouping, resampling, missing value handling, outlier treatment.
    """

    # Step 1: Group and Resample
    grouped_resampled_data = group_and_resample_time_series(
        df, date_column, target_column, frequency, group_by_columns
    )

    processed_groups = {}
    for group_key, group_df in grouped_resampled_data.items():
        # Step 2: Handle Missing Values
        group_df_filled = handle_missing_values(group_df, target_column, method=missing_value_method)

        # Step 3: Handle Outliers (optional)
        if outlier_treatment_method == 'iqr':
            group_df_treated = detect_and_treat_outliers_iqr(group_df_filled, target_column)
        else:
            group_df_treated = group_df_filled # No outlier treatment

        # Ensure the target column is still present and correctly named
        if target_column not in group_df_treated.columns:
            # This might happen if the target column was the only column and got renamed implicitly
            # or if an operation removed it. This check is a safeguard.
            if len(group_df_treated.columns) == 1:
                 group_df_treated.columns = [target_column]
            else:
                raise Exception(f"Target column '{target_column}' lost during processing of group {group_key}")

        processed_groups[group_key] = group_df_treated

    return processed_groups


# Example usage (for testing purposes)
if __name__ == '__main__':
    data = {
        'x_ResourceGroupName': ['RG1', 'RG1', 'RG2', 'RG1', 'RG2', 'RG1', 'RG1', 'RG2'],
        'ChargePeriodStart': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03', '2023-01-02', '2023-01-01', '2023-01-10', '2023-01-08'], # Added more data for resampling
        'TotalBilledCostInUSD': [10, 12, 5, 11, 6, 100, 15, 50], # Added outliers
        'Subscription': ['Sub1', 'Sub1', 'Sub2', 'Sub1', 'Sub2', 'Sub1', 'Sub1', 'Sub2']
    }
    sample_df = pd.DataFrame(data)

    # Make a copy for parsing date column to avoid SettingWithCopyWarning in direct use
    df_to_process = sample_df.copy()

    # Ensure ChargePeriodStart is datetime
    df_to_process['ChargePeriodStart'] = pd.to_datetime(df_to_process['ChargePeriodStart'])


    print("Original DataFrame:")
    print(df_to_process)
    print("\n" + "="*30 + "\n")

    # Test case 1: Grouping by ResourceGroupName, daily frequency
    try:
        print("Test Case 1: Group by ResourceGroupName, Daily, Sum, ffill, iqr")
        processed_data_rg = full_preprocess_pipeline(
            df_to_process,
            date_column='ChargePeriodStart',
            target_column='TotalBilledCostInUSD',
            frequency='D',
            group_by_columns=['x_ResourceGroupName'],
            missing_value_method='interpolate', # Use interpolate for better handling of internal NaNs
            outlier_treatment_method='iqr'
        )
        for group_name, group_df in processed_data_rg.items():
            print(f"\n--- Group: {group_name} ---")
            print(group_df)
            # Ensure target column is correctly named
            assert 'TotalBilledCostInUSD' in group_df.columns, f"Target column missing in group {group_name}"
            assert isinstance(group_df.index, pd.DatetimeIndex), f"Index is not DatetimeIndex in group {group_name}"

    except ValueError as ve:
        print(f"ValueError in Test Case 1: {ve}")
    except Exception as e:
        print(f"Error in Test Case 1: {e}")

    print("\n" + "="*30 + "\n")

    # Test case 2: No grouping, weekly frequency
    try:
        print("Test Case 2: No Grouping, Weekly, Sum, ffill, iqr")
        processed_data_overall_w = full_preprocess_pipeline(
            df_to_process.copy(), # Use a copy to avoid side effects if any
            date_column='ChargePeriodStart',
            target_column='TotalBilledCostInUSD',
            frequency='W',
            group_by_columns=None,
            missing_value_method='zero',
            outlier_treatment_method='iqr'
        )
        for group_name, group_df in processed_data_overall_w.items():
            print(f"\n--- Group: {group_name} ---")
            print(group_df)
            assert 'TotalBilledCostInUSD' in group_df.columns, f"Target column missing in group {group_name}"
            assert isinstance(group_df.index, pd.DatetimeIndex), f"Index is not DatetimeIndex in group {group_name}"

    except ValueError as ve:
        print(f"ValueError in Test Case 2: {ve}")
    except Exception as e:
        print(f"Error in Test Case 2: {e}")

    print("\n" + "="*30 + "\n")

    # Test case 3: Grouping by multiple columns, monthly frequency
    try:
        print("Test Case 3: Group by ResourceGroupName and Subscription, Monthly, Sum, median, no outlier treatment")
        processed_data_multi_group_m = full_preprocess_pipeline(
            df_to_process.copy(),
            date_column='ChargePeriodStart',
            target_column='TotalBilledCostInUSD',
            frequency='M',
            group_by_columns=['x_ResourceGroupName', 'Subscription'],
            missing_value_method='median',
            outlier_treatment_method=None
        )
        for group_name, group_df in processed_data_multi_group_m.items():
            print(f"\n--- Group: {group_name} ---") # group_name will be a tuple
            print(group_df)
            assert 'TotalBilledCostInUSD' in group_df.columns, f"Target column missing in group {group_name}"
            assert isinstance(group_df.index, pd.DatetimeIndex), f"Index is not DatetimeIndex in group {group_name}"

    except ValueError as ve:
        print(f"ValueError in Test Case 3: {ve}")
    except Exception as e:
        print(f"Error in Test Case 3: {e}")
