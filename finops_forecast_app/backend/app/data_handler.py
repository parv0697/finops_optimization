import pandas as pd
from typing import List, Optional, Tuple, Dict, Union

def load_csv_from_path(file_path: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame more robustly.
    Tries common encodings if default utf-8 fails.
    """
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            raise ValueError(f"Error loading CSV: Could not decode file with common encodings. Original error: {e}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")


def get_column_names(df: pd.DataFrame) -> List[str]:
    """Returns a list of column names from the DataFrame."""
    return df.columns.tolist()


def identify_date_column(df: pd.DataFrame, date_col_name: Optional[str] = None) -> str:
    """
    Identifies or validates the date column.
    If date_col_name is provided, it validates it.
    Otherwise, it tries to infer it (basic inference).
    Returns the name of the date column.
    """
    if date_col_name:
        if date_col_name not in df.columns:
            raise ValueError(f"Specified date column '{date_col_name}' not found in DataFrame.")
        # Attempt to convert to datetime to validate if it's date-like
        try:
            pd.to_datetime(df[date_col_name], errors='raise')
            return date_col_name
        except Exception as e:
            raise ValueError(f"Column '{date_col_name}' could not be parsed as a date series: {e}")
    else:
        # Basic inference: look for columns with 'date' or 'time' in their name
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'period' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='raise')
                    return col # Return first column that successfully parses
                except Exception:
                    continue
        raise ValueError("Could not automatically infer a date column. Please specify one.")


def identify_target_column(df: pd.DataFrame, target_col_name: str) -> str:
    """
    Validates the target column and ensures it's numeric.
    Returns the name of the target column.
    """
    if not target_col_name:
        raise ValueError("Target column name must be specified.")
    if target_col_name not in df.columns:
        raise ValueError(f"Specified target column '{target_col_name}' not found in DataFrame.")

    # Check if target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col_name]):
        # Try to convert to numeric if it's object type, might be numbers stored as strings
        try:
            df[target_col_name] = pd.to_numeric(df[target_col_name])
        except ValueError as e:
            raise ValueError(f"Target column '{target_col_name}' is not numeric and could not be converted to numeric: {e}")

    return target_col_name


def parse_date_column(df: pd.DataFrame, date_column_name: str) -> pd.DataFrame:
    """
    Parses the specified date column to datetime objects.
    Handles potential errors during parsing.
    """
    if date_column_name not in df.columns:
        raise ValueError(f"Date column '{date_column_name}' not found.")

    try:
        # Attempt to infer datetime format for robustness, but can be slow.
        # For faster parsing, if format is known, pass it to to_datetime.
        df[date_column_name] = pd.to_datetime(df[date_column_name], infer_datetime_format=True)
    except Exception as e:
        # If specific formats are common, try them here.
        # Example: df[date_column_name] = pd.to_datetime(df[date_column_name], format='%Y-%m-%d')
        raise ValueError(f"Could not parse date column '{date_column_name}'. Error: {e}. Ensure dates are in a consistent, recognizable format.")

    return df

# Example usage (for testing purposes, not part of the module's API):
if __name__ == '__main__':
    # Create a sample DataFrame
    data = {
        'x_ResourceGroupName': ['RG1', 'RG1', 'RG2', 'RG1', 'RG2'],
        'ChargePeriodStart': ['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-03', '2023-01-02'],
        'TotalBilledCostInUSD': [10, 12, 5, 11, 6],
        'Subscription': ['Sub1', 'Sub1', 'Sub2', 'Sub1', 'Sub2'],
        'RandomCol': [1,2,3,4,5]
    }
    sample_df = pd.DataFrame(data)
    sample_df.to_csv('sample_data_temp.csv', index=False)

    try:
        df_loaded = load_csv_from_path('sample_data_temp.csv')
        print("Loaded DataFrame:")
        print(df_loaded.head())
        print("\nColumn Names:", get_column_names(df_loaded))

        date_col = identify_date_column(df_loaded, 'ChargePeriodStart')
        # date_col_auto = identify_date_column(df_loaded) # Test auto-inference
        # print(f"Identified date column (specified): {date_col}")
        # print(f"Identified date column (auto): {date_col_auto}")


        target_col = identify_target_column(df_loaded, 'TotalBilledCostInUSD')
        print(f"Identified target column: {target_col}")

        df_parsed = parse_date_column(df_loaded, date_col)
        print(f"\nParsed date column '{date_col}':")
        print(df_parsed.info())
        print(df_parsed[date_col].head())

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        import os
        if os.path.exists('sample_data_temp.csv'):
            os.remove('sample_data_temp.csv')
