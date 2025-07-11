import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import warnings

# Suppress warnings from statsmodels and pmdarima to keep output clean
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=UserWarning, module="pmdarima")

class BaseForecastingModel(ABC):
    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.model = None
        self.model_params = model_params if model_params else {}
        self.train_summary = None # To store any summary from training

    @abstractmethod
    def train(self, series: pd.Series, exog: Optional[pd.DataFrame] = None):
        """
        Trains the forecasting model.
        Args:
            series (pd.Series): The time series data to train on (endogenous variable).
                               Must have a DatetimeIndex.
            exog (Optional[pd.DataFrame]): Exogenous variables, if the model supports them.
                                          Must have a DatetimeIndex aligned with `series`.
        """
        pass

    @abstractmethod
    def predict(self, n_periods: int, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generates forecasts for n_periods.
        Args:
            n_periods (int): The number of future periods to forecast.
            exog (Optional[pd.DataFrame]): Future exogenous variables, if required by the model.
                                          Index should cover the forecast period.
        Returns:
            pd.Series: A series of forecasted values with a DatetimeIndex.
        """
        pass

    def get_train_summary(self) -> Optional[str]:
        """Returns a string summary of the trained model, if available."""
        return str(self.train_summary) if self.train_summary else "No training summary available."


class ARIMAModel(BaseForecastingModel):
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__("ARIMA", model_params)
        # Default pmdarima auto_arima parameters - can be overridden by model_params
        self.auto_arima_params = {
            'start_p': 1, 'start_q': 1,
            'max_p': 3, 'max_q': 3, 'm': 1, # m=1 for non-seasonal, adjust if seasonality is known
            'seasonal': False, # Set to True and specify m if data is seasonal
            'stepwise': True, 'suppress_warnings': True,
            'D': None, 'max_D': 1,
            'error_action': 'ignore',
            'trace': False # Set to True for detailed output during fitting
        }
        if model_params:
            self.auto_arima_params.update(model_params)


    def train(self, series: pd.Series, exog: Optional[pd.DataFrame] = None):
        import pmdarima as pm

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("ARIMA model requires the input series to have a DatetimeIndex.")

        # pmdarima handles NaNs by default, but it's better to ensure clean data
        series_clean = series.dropna()
        if series_clean.empty:
            raise ValueError("Input series is empty after dropping NaNs.")

        # auto_arima parameters can be passed via self.auto_arima_params
        # Ensure exog is correctly aligned and passed if provided
        current_exog = None
        if exog is not None:
            if not isinstance(exog.index, pd.DatetimeIndex):
                 raise ValueError("Exogenous data for ARIMA must have a DatetimeIndex.")
            # Align exog with series_clean
            current_exog = exog.reindex(series_clean.index)
            # pmdarima might require exog to not have NaNs for rows corresponding to series_clean
            if current_exog.isnull().any().any():
                 warnings.warn(f"NaNs found in exogenous data for {self.model_name} training period. Applying ffill and bfill.")
                 current_exog = current_exog.fillna(method='ffill').fillna(method='bfill')
                 if current_exog.isnull().any().any(): # If still NaNs (e.g. all NaNs column)
                     raise ValueError("Exogenous data still contains NaNs after fill for ARIMA training.")


        self.model = pm.auto_arima(
            series_clean,
            X=current_exog, # Pass aligned exogenous variables
            **self.auto_arima_params
        )
        self.train_summary = self.model.summary()
        # print(f"{self.model_name} model trained. Best order: {self.model.order}, Seasonal order: {self.model.seasonal_order if self.auto_arima_params.get('seasonal', False) else 'N/A'}")
        self.train_summary = self.model.summary().as_text()


    def predict(self, n_periods: int, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained yet.")

        future_exog = None
        if exog is not None:
            if not isinstance(exog.index, pd.DatetimeIndex):
                 raise ValueError("Future exogenous data for ARIMA must have a DatetimeIndex.")
            if len(exog) < n_periods:
                raise ValueError(f"Future exogenous data must cover all {n_periods} forecast periods.")
            future_exog = exog.head(n_periods)
            if future_exog.isnull().any().any():
                 warnings.warn(f"NaNs found in future exogenous data for {self.model_name} prediction. Applying ffill and bfill.")
                 future_exog = future_exog.fillna(method='ffill').fillna(method='bfill')
                 if future_exog.isnull().any().any():
                     raise ValueError("Future exogenous data still contains NaNs after fill for ARIMA prediction.")

        forecast_values, conf_int = self.model.predict(
            n_periods=n_periods,
            X=future_exog, # Pass future exogenous variables
            return_conf_int=True
        )

        # Create DatetimeIndex for the forecast
        last_date = self.model.arima_res_.data.endog.index[-1] # Get last date from original series used in fit
        freq = pd.infer_freq(self.model.arima_res_.data.endog.index) # Infer frequency
        if freq is None: # Fallback if freq cannot be inferred (e.g. too few points or irregular)
            # Attempt to get freq from series passed to train, if available and different
            # This can be complex. A simpler fallback:
            if hasattr(self.model.arima_res_.data.endog.index, 'freqstr') and self.model.arima_res_.data.endog.index.freqstr:
                freq = self.model.arima_res_.data.endog.index.freqstr
            else: # Default to Day if really stuck, or require freq passed in
                freq = 'D'
                warnings.warn(f"Could not infer frequency for ARIMA forecast index; defaulting to '{freq}'. Ensure training series has a clear frequency.")

        forecast_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=n_periods, freq=freq)

        return pd.Series(forecast_values, index=forecast_index, name=f"{self.model_name}_forecast")


class ProphetModel(BaseForecastingModel):
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__("Prophet", model_params)
        # Prophet default parameters can be specified here or passed via model_params
        # e.g., self.prophet_params = {'seasonality_mode': 'multiplicative'}
        # self.prophet_params.update(model_params)

    def train(self, series: pd.Series, exog: Optional[pd.DataFrame] = None):
        from prophet import Prophet

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Prophet model requires the input series to have a DatetimeIndex.")

        df_train = series.reset_index()
        df_train.columns = ['ds', 'y']
        df_train['ds'] = pd.to_datetime(df_train['ds'])

        # Handle exogenous variables
        if exog is not None:
            if not isinstance(exog.index, pd.DatetimeIndex):
                raise ValueError("Exogenous data for Prophet must have a DatetimeIndex.")
            # Ensure exog index is 'ds' and aligned
            exog_prophet = exog.copy()
            exog_prophet.index.name = 'ds'
            exog_prophet = exog_prophet.reset_index()
            df_train = pd.merge(df_train, exog_prophet, on='ds', how='left')
            # Prophet handles NaNs in regressors by not using them for that row,
            # but it's better if they are filled if that makes sense for the variable.
            # For simplicity, we assume user handles NaNs in exog before passing, or Prophet's default is fine.


        self.model = Prophet(**self.model_params)

        if exog is not None:
            for col in exog.columns:
                self.model.add_regressor(col)

        self.model.fit(df_train)
        # Prophet doesn't have a simple text summary like statsmodels.
        # self.train_summary = "Prophet model fitted."
        # Could potentially store info about detected seasonalities if verbose=True during fit,
        # or parameters if `self.model.params` is inspected. For now, keep it simple.

    def predict(self, n_periods: int, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained yet.")

        future_df = self.model.make_future_dataframe(periods=n_periods, freq=self.model.stan_fit['freq'])

        # Handle future exogenous variables
        if exog is not None:
            if not isinstance(exog.index, pd.DatetimeIndex):
                raise ValueError("Future exogenous data for Prophet must have a DatetimeIndex.")
            if len(exog) < n_periods:
                raise ValueError(f"Future exogenous data must cover all {n_periods} forecast periods for Prophet.")

            exog_future = exog.copy().head(n_periods) # Ensure it matches the n_periods exactly from start of forecast range
            exog_future.index.name = 'ds'
            exog_future = exog_future.reset_index()

            # Merge exog with future_df. Prophet expects regressors in the df passed to predict.
            # The future_df already has the 'ds' column for the future dates.
            # We need to align exog_future to these dates.
            # Assuming exog_future's index starts correctly after the training data.
            # A robust way is to ensure `future_df` (which has the correct future 'ds' dates) gets these exog values.
            if not future_df['ds'].equals(exog_future['ds']): # Basic check, might need more robust alignment
                 # This merge is crucial: future_df provides the dates, exog_future provides regressor values for those dates.
                 # We only care about the future dates, so slice future_df to only include forecast period.
                future_dates_only = future_df[future_df['ds'] > self.model.history_dates.max()].copy()
                # Clean exog_future from any past dates if necessary, ensure it's just for the forecast horizon
                exog_future = exog_future[exog_future['ds'].isin(future_dates_only['ds'])]

                # Now merge exog_future into future_dates_only
                # If exog_future was prepared correctly (indexed by future dates), we can assign:
                # For each column in exog_future (excluding 'ds'), add it to future_dates_only

                # Safer merge:
                temp_future_df = pd.merge(future_dates_only[['ds']], exog_future, on='ds', how='left')

                # Fill any NaNs in regressors for future dates (e.g., using ffill from last known exog value or mean)
                # Prophet will error if a regressor column added with add_regressor is all NaN in future_df.
                for col in self.model.extra_regressors.keys():
                    if col in temp_future_df.columns and temp_future_df[col].isnull().any():
                        warnings.warn(f"NaNs found in future regressor '{col}' for Prophet. Filling with ffill then bfill.")
                        temp_future_df[col] = temp_future_df[col].fillna(method='ffill').fillna(method='bfill')
                        if temp_future_df[col].isnull().all(): # If still all NaNs after fill (e.g. exog was empty)
                            temp_future_df[col] = 0 # Fill with 0 as a last resort
                            warnings.warn(f"Future regressor '{col}' was all NaNs after fill, setting to 0 for Prophet prediction.")


                # Update the original future_df (which includes history) with these future regressor values
                # This is a bit tricky. Prophet's `predict` uses the `future_df` that `make_future_dataframe` creates.
                # This df can include historical dates if `include_history=True` (default).
                # We need to ensure the regressor columns are populated for the future part of this df.

                # Let's reconstruct future_df for prediction carefully
                # If self.model.extra_regressors is not empty:
                if self.model.extra_regressors:
                    # Start with just the 'ds' column for future dates
                    future_ds = self.model.make_future_dataframe(periods=n_periods, freq=self.model.stan_fit['freq'], include_history=False)

                    # Merge future exogenous variables
                    # exog should be prepared to have 'ds' and regressor columns, covering the future_ds dates
                    exog_for_join = exog.copy()
                    exog_for_join.index.name = 'ds'
                    exog_for_join = exog_for_join.reset_index()

                    future_df_with_exog = pd.merge(future_ds, exog_for_join, on='ds', how='left')

                    # Fill NaNs in regressors as discussed
                    for col in self.model.extra_regressors.keys():
                        if col in future_df_with_exog.columns and future_df_with_exog[col].isnull().any():
                            future_df_with_exog[col] = future_df_with_exog[col].fillna(method='ffill').fillna(method='bfill').fillna(0)

                    # If include_history was True for original future_df, need to handle that.
                    # For simplicity here, we'll make future_df for prediction only for future dates if exog is used.
                    # This means forecast will not include historical fit if exog is present.
                    # This is often fine as we usually care about future predictions.
                    # If historical fit with exog is needed, train_df should be part of this future_df_with_exog.
                    final_future_df_for_predict = future_df_with_exog
                else:
                    final_future_df_for_predict = future_df # The one from make_future_dataframe

            else: # No exog
                 final_future_df_for_predict = future_df


        forecast = self.model.predict(final_future_df_for_predict)

        # Return only the forecast for future periods
        # `forecast` contains historical fit and future predictions.
        # `forecast['ds']` are the dates. We need to pick out the future ones.
        # The `future_df` (or `final_future_df_for_predict`) generated by `make_future_dataframe`
        # already contains the correct future dates. We need `n_periods` from the end.

        # Get the last `n_periods` from the forecast result.
        # These correspond to the `n_periods` future dates.
        forecast_values = forecast['yhat'].iloc[-n_periods:]
        forecast_index = forecast['ds'].iloc[-n_periods:]

        return pd.Series(forecast_values.values, index=pd.to_datetime(forecast_index.values), name=f"{self.model_name}_forecast")

# Factory function to get model instance
def get_model(model_name: str, model_params: Optional[Dict[str, Any]] = None) -> BaseForecastingModel:
    if model_name.upper() == "ARIMA":
        return ARIMAModel(model_params=model_params)
    elif model_name.upper() == "PROPHET":
        return ProphetModel(model_params=model_params)
    # Add other models here:
    elif model_name.upper() == "LSTM":
        return LSTMModel(model_params=model_params)
    elif model_name.upper() == "EXPONENTIALSMOOTHING":
        return ExponentialSmoothingModel(model_params=model_params)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")


if __name__ == '__main__':
    # Create Sample Time Series Data
    date_rng_train = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    data_train = pd.Series(range(len(date_rng_train)) + 10*pd.np.random.rand(len(date_rng_train)), index=date_rng_train)

    # Sample exogenous data (optional)
    exog_data_train = pd.DataFrame({
        'holiday': pd.Series(0, index=data_train.index).sample(frac=0.1, replace=True, random_state=1).apply(lambda x: 1 if x > 0 else 0), # random holidays
        'marketing_spend': pd.Series(pd.np.random.rand(len(data_train)) * 100, index=data_train.index)
    })

    # Future exogenous data for prediction period
    n_forecast_periods = 12
    last_train_date = data_train.index[-1]
    future_date_rng = pd.date_range(start=last_train_date + pd.DateOffset(months=1), periods=n_forecast_periods, freq='M')
    exog_data_future = pd.DataFrame({
        'holiday': pd.Series(0, index=future_date_rng), # Assuming no holidays in future for simplicity or actual known future holidays
        'marketing_spend': pd.Series(pd.np.random.rand(len(future_date_rng)) * 110, index=future_date_rng) # Slightly increased spend
    })

    print("Sample Training Data Head:")
    print(data_train.head())
    if exog_data_train is not None:
        print("\nSample Exogenous Training Data Head:")
        print(exog_data_train.head())
        print("\nSample Exogenous Future Data Head:")
        print(exog_data_future.head())

    # --- Test ARIMA ---
    print("\n--- Testing ARIMA Model ---")
    try:
        arima_model = get_model("ARIMA")
        # Test without exog
        # arima_model.train(data_train)
        # arima_forecast = arima_model.predict(n_periods=n_forecast_periods)
        # print("\nARIMA Forecast (no exog):")
        # print(arima_forecast)
        # print(arima_model.get_train_summary())

        # Test with exog
        print("\nARIMA with Exog:")
        arima_model_exog = get_model("ARIMA", model_params={'seasonal': False}) # Assuming non-seasonal for this example exog
        arima_model_exog.train(data_train, exog=exog_data_train)
        arima_forecast_exog = arima_model_exog.predict(n_periods=n_forecast_periods, exog=exog_data_future)
        print("\nARIMA Forecast (with exog):")
        print(arima_forecast_exog)
        # print(arima_model_exog.get_train_summary())


    except Exception as e:
        print(f"Error testing ARIMA: {e}")
        import traceback
        traceback.print_exc()

    # --- Test Exponential Smoothing ---
    print("\n--- Testing Exponential Smoothing Model ---")
    try:
        # Simple test (additive trend, no seasonality)
        es_model_simple = get_model("ExponentialSmoothing", model_params={'trend': 'add', 'seasonal': None})
        es_model_simple.train(data_train)
        es_forecast_simple = es_model_simple.predict(n_periods=n_forecast_periods)
        print("\nExponential Smoothing Forecast (Simple - Additive Trend):")
        print(es_forecast_simple)
        # print(es_model_simple.get_train_summary())

        # Seasonal test
        # Need enough data and appropriate seasonal_periods for this to work well.
        # Our sample data is monthly, so seasonal_periods=12 for yearly seasonality.
        if len(data_train) >= 24 : # Need at least 2 full seasons
            es_model_seasonal = get_model("ExponentialSmoothing", model_params={'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12})
            es_model_seasonal.train(data_train)
            es_forecast_seasonal = es_model_seasonal.predict(n_periods=n_forecast_periods)
            print("\nExponential Smoothing Forecast (Additive Trend, Additive Seasonality M=12):")
            print(es_forecast_seasonal)
            # print(es_model_seasonal.get_train_summary())
        else:
            print("\nSkipping seasonal Exponential Smoothing test, data too short for seasonal_periods=12.")

    except Exception as e:
        print(f"Error testing Exponential Smoothing: {e}")
        import traceback
        traceback.print_exc()

    # --- Test LSTM Model ---
    print("\n--- Testing LSTM Model ---")
    try:
        # LSTM requires more data points typically.
        # Our monthly data_train has 48 points. n_steps=12 means 1 year lookback.
        if len(data_train) > 24: # Arbitrary minimum length for a basic LSTM test
            lstm_model = get_model("LSTM", model_params={'n_steps': 6, 'n_features': 1, 'epochs': 50, 'verbose': 0})
            lstm_model.train(data_train) # LSTM train method needs to handle series
            lstm_forecast = lstm_model.predict(n_periods=n_forecast_periods)
            print("\nLSTM Forecast:")
            print(lstm_forecast)
            # LSTM model does not have a text summary like statsmodels.
            print(lstm_model.get_train_summary())
        else:
            print("\nSkipping LSTM test, data too short for meaningful n_steps.")

    except Exception as e:
        print(f"Error testing LSTM: {e}")
        import traceback
        traceback.print_exc()


# LSTM Model Implementation
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Conditionally import TensorFlow to avoid error if not installed and not used.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not found. LSTMModel will not be available.")


class LSTMModel(BaseForecastingModel):
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__("LSTM", model_params)
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMModel but not installed.")

        # Default LSTM parameters
        self.n_steps = model_params.get('n_steps', 12) # Look-back window
        self.n_features = model_params.get('n_features', 1) # Univariate
        self.epochs = model_params.get('epochs', 200)
        self.verbose = model_params.get('verbose', 0) # Keras verbose level
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history_data_for_pred = None # Store last n_steps of training data for prediction

    def _create_sequences(self, data, n_steps):
        X, y = [], []
        for i in range(len(data)):
            end_ix = i + n_steps
            if end_ix >= len(data):
                break
            seq_x, seq_y = data[i:end_ix], data[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    def train(self, series: pd.Series, exog: Optional[pd.DataFrame] = None):
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("LSTM model requires the input series to have a DatetimeIndex.")
        if exog is not None:
            warnings.warn("Current LSTMModel implementation is univariate and will ignore exogenous variables.")

        # Ensure series is numpy array of shape (n_samples, 1) for scaler
        data_values = series.dropna().values.reshape(-1, 1)
        if len(data_values) < self.n_steps + 1: # Need at least n_steps for input and 1 for output
            raise ValueError(f"Series too short for LSTM with n_steps={self.n_steps}. Need at least {self.n_steps + 1} data points.")

        scaled_data = self.scaler.fit_transform(data_values)

        X, y = self._create_sequences(scaled_data, self.n_steps)
        if X.shape[0] == 0 : # Not enough data to form any sequence
             raise ValueError(f"Not enough data to create sequences for LSTM with n_steps={self.n_steps}. Series length after dropna: {len(data_values)}")

        # Reshape X to [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))

        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(self.n_steps, self.n_features))) # 50 LSTM units
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')

        # Store the last n_steps of scaled training data to initiate prediction sequence
        self.history_data_for_pred = scaled_data[-self.n_steps:].reshape((1, self.n_steps, self.n_features))
        self.last_train_date = series.index[-1] # Store last date for forecast index
        self.train_freq = pd.infer_freq(series.index) # Store frequency for forecast index

        history = self.model.fit(X, y, epochs=self.epochs, verbose=self.verbose)
        self.train_summary = f"LSTM model trained for {self.epochs} epochs. Final loss: {history.history['loss'][-1]:.4f}"

    def predict(self, n_periods: int, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        if self.model is None or self.history_data_for_pred is None:
            raise ValueError("Model not trained yet or history_data_for_pred is missing.")
        if exog is not None:
            warnings.warn("Current LSTMModel implementation is univariate and will ignore exogenous variables for prediction.")

        temp_input = list(self.history_data_for_pred.flatten()) # Start with the last sequence from training
        lst_output = []

        for _ in range(n_periods):
            if len(temp_input) > self.n_steps:
                x_input = np.array(temp_input[1:]) # Get last n_steps
            else:
                x_input = np.array(temp_input)

            x_input = x_input.reshape((1, self.n_steps, self.n_features))
            yhat = self.model.predict(x_input, verbose=0)

            temp_input.append(yhat[0,0]) # Append the prediction (scalar)
            temp_input = temp_input[1:] # Roll the window
            lst_output.append(yhat[0,0])

        forecast_scaled = np.array(lst_output).reshape(-1, 1)
        forecast_values = self.scaler.inverse_transform(forecast_scaled).flatten()

        # Create DatetimeIndex for the forecast
        if self.train_freq is None:
            self.train_freq = 'D' # Default if somehow not inferred
            warnings.warn(f"Could not infer frequency for LSTM forecast index; defaulting to '{self.train_freq}'.")

        forecast_index = pd.date_range(
            start=self.last_train_date + pd.tseries.frequencies.to_offset(self.train_freq),
            periods=n_periods,
            freq=self.train_freq
        )
        return pd.Series(forecast_values, index=forecast_index, name=f"{self.model_name}_forecast")


class ExponentialSmoothingModel(BaseForecastingModel):
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        super().__init__("ExponentialSmoothing", model_params)
        # Default parameters for ExponentialSmoothing
        # Common params: trend ('add', 'mul', None), seasonal ('add', 'mul', None), seasonal_periods (int)
        # damped_trend (bool)
        self.es_params = {
            'trend': None,
            'seasonal': None,
            'seasonal_periods': None, # Must be provided if seasonal is not None
            'damped_trend': False,
            'initialization_method': 'estimated', # Default is 'estimated'
            # 'use_boxcox': False, # Can be True, 'log', or a float lambda
            # 'optimized': True # Default, let statsmodels optimize parameters
        }
        if model_params:
            self.es_params.update(model_params)
            # Ensure seasonal_periods is provided if seasonality is requested
            if self.es_params.get('seasonal') and not self.es_params.get('seasonal_periods'):
                # Try to infer seasonal_periods from series frequency if possible, or default
                # This inference is tricky here without the series. Better to require it in params.
                # For now, we'll rely on user passing it or statsmodels defaulting/erroring.
                pass # Example: if freq is 'M', seasonal_periods could be 12 for yearly.

    def train(self, series: pd.Series, exog: Optional[pd.DataFrame] = None):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError(f"{self.model_name} model requires the input series to have a DatetimeIndex.")
        if exog is not None:
            warnings.warn(f"{self.model_name} from statsmodels does not directly support exogenous variables in the same way as ARIMA/SARIMAX. Exog will be ignored.")

        series_clean = series.dropna()
        if series_clean.empty:
            raise ValueError("Input series is empty after dropping NaNs for ExponentialSmoothing.")

        # Handle cases where series might be too short for certain seasonal_periods
        min_len_for_seasonal = 2 * self.es_params.get('seasonal_periods', 0) if self.es_params.get('seasonal') else 0
        if self.es_params.get('seasonal') and len(series_clean) < min_len_for_seasonal :
            warnings.warn(f"Series too short for seasonal_periods={self.es_params['seasonal_periods']}. Fitting non-seasonal ExponentialSmoothing instead.")
            current_params = self.es_params.copy()
            current_params['seasonal'] = None
            current_params['seasonal_periods'] = None
        else:
            current_params = self.es_params

        # Statsmodels ExponentialSmoothing expects endog to be 1-d.
        self.model = ExponentialSmoothing(
            series_clean,
            **current_params
        ).fit()
        self.train_summary = self.model.summary().as_text()

    def predict(self, n_periods: int, exog: Optional[pd.DataFrame] = None) -> pd.Series:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        if exog is not None:
            warnings.warn(f"{self.model_name} does not use exogenous variables for prediction. Exog will be ignored.")

        # Statsmodels forecast method for ExponentialSmoothing returns a series with DatetimeIndex
        forecast_values = self.model.forecast(steps=n_periods)

        return pd.Series(forecast_values, name=f"{self.model_name}_forecast")


if __name__ == '__main__':
    # Create Sample Time Series Data
    date_rng_train = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')


    # --- Test Prophet ---
    print("\n--- Testing Prophet Model ---")
    try:
        prophet_model = get_model("Prophet")
        # Test without exog
        # prophet_model.train(data_train)
        # prophet_forecast = prophet_model.predict(n_periods=n_forecast_periods)
        # print("\nProphet Forecast (no exog):")
        # print(prophet_forecast)

        # Test with exog
        print("\nProphet with Exog:")
        prophet_model_exog = get_model("Prophet")
        prophet_model_exog.train(data_train, exog=exog_data_train)
        prophet_forecast_exog = prophet_model_exog.predict(n_periods=n_forecast_periods, exog=exog_data_future)
        print("\nProphet Forecast (with exog):")
        print(prophet_forecast_exog)

    except Exception as e:
        print(f"Error testing Prophet: {e}")
        import traceback
        traceback.print_exc()
