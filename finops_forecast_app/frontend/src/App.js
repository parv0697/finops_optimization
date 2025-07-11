import React, { useState, useEffect } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ColumnSelector from './components/ColumnSelector';
import ModelSelector from './components/ModelSelector';
import ResultsDisplay from './components/ResultsDisplay'; // Import ResultsDisplay
import apiClient from './services/apiService';

function App() {
  const [uploadedFilePath, setUploadedFilePath] = useState(null);
  const [originalFileName, setOriginalFileName] = useState('');
  const [columnNames, setColumnNames] = useState([]);
  const [isLoadingColumns, setIsLoadingColumns] = useState(false);
  const [error, setError] = useState('');

  // Preprocessing Step State
  const [dateColumn, setDateColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  const [groupByColumns, setGroupByColumns] = useState([]);
  const [frequency, setFrequency] = useState('D'); // Default to Daily

  // Model Selection State
  const [selectedModels, setSelectedModels] = useState(['ARIMA', 'Prophet', 'ExponentialSmoothing']); // Default selection

  // Results State
  const [preprocessingRunId, setPreprocessingRunId] = useState(null);
  const [forecastResults, setForecastResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);


  const handleUploadSuccess = async (filepath, filename) => {
    setUploadedFilePath(filepath);
    setOriginalFileName(filename);
    setError('');
    setIsLoadingColumns(true);
    try {
      const response = await apiClient.post('/get_column_names', { filepath });
      setColumnNames(response.data.columns || []);
      if ((response.data.columns || []).length === 0) {
        setError('No columns found in the uploaded file or failed to parse.');
      }
    } catch (err) {
      let errorMessage = 'Error fetching column names.';
      if (err.response) {
        errorMessage = `Error: ${err.response.data.detail || err.response.statusText}`;
      } else if (err.request) {
        errorMessage = 'Error: No response from server when fetching columns.';
      } else {
        errorMessage = `Error: ${err.message}`;
      }
      setError(errorMessage);
      setColumnNames([]);
    } finally {
      setIsLoadingColumns(false);
    }
  };

  const handleUploadError = (errorMessage) => {
    setUploadedFilePath(null);
    setOriginalFileName('');
    setColumnNames([]);
    setError(errorMessage);
  };

  const handlePreprocessingSubmit = async () => {
    if (!uploadedFilePath || !dateColumn || !targetColumn || !frequency) {
        setError("Please ensure a file is uploaded and date column, target column, and frequency are selected.");
        return;
    }
    setIsProcessing(true);
    setError('');
    setPreprocessingRunId(null);
    setForecastResults(null);

    const preprocessParams = {
        filepath: uploadedFilePath,
        date_column: dateColumn,
        target_column: targetColumn,
        frequency: frequency,
        group_by_columns: groupByColumns.length > 0 ? groupByColumns : null,
        // Using default missing_value_method and outlier_treatment_method for now
    };

    try {
        const response = await apiClient.post('/preprocess_data', preprocessParams);
        setPreprocessingRunId(response.data.preprocessing_run_id);
        // TODO: Display response.data.summary_by_group if needed
        setError(''); // Clear previous errors
        alert(`Preprocessing successful! Run ID: ${response.data.preprocessing_run_id}`);
    } catch (err) {
        let errorMessage = 'Error during preprocessing.';
        if (err.response) {
            errorMessage = `Preprocessing Error: ${err.response.data.detail || err.response.statusText}`;
        } else if (err.request) {
            errorMessage = 'Preprocessing Error: No response from server.';
        } else {
            errorMessage = `Preprocessing Error: ${err.message}`;
        }
        setError(errorMessage);
        setPreprocessingRunId(null);
    } finally {
        setIsProcessing(false);
    }
  };

  const handleRunForecasting = async () => {
    if (!preprocessingRunId || !targetColumn) {
        setError("Preprocessing must be completed successfully first, and target column must be known.");
        return;
    }
    setIsProcessing(true);
    setError('');
    setForecastResults(null);

    const forecastParams = {
        preprocessing_run_id: preprocessingRunId,
        target_column: targetColumn, // Crucial for identifying target in processed parquet
        models_to_train: selectedModels,
        forecast_horizon: 30, // Default, make this configurable later
        // evaluation_split_ratio and model_params can be added later
    };

    try {
        const response = await apiClient.post('/train_forecast_evaluate', forecastParams);
        setForecastResults(response.data);
        // TODO: Pass this to a results display component
        alert('Forecasting complete! Check console for results (for now).');
        console.log("Forecast Results:", response.data);
    } catch (err) {
        let errorMessage = 'Error during forecasting.';
        if (err.response) {
            errorMessage = `Forecasting Error: ${err.response.data.detail || err.response.statusText}`;
        } else if (err.request) {
            errorMessage = 'Forecasting Error: No response from server.';
        } else {
            errorMessage = `Forecasting Error: ${err.message}`;
        }
        setError(errorMessage);
        setForecastResults(null);
    } finally {
        setIsProcessing(false);
    }
  };


  return (
    <div className="App">
      <header className="App-header">
        <h1>FinOps Cloud Cost Forecaster</h1>
      </header>

      <FileUpload
        onUploadSuccess={handleUploadSuccess}
        onUploadError={handleUploadError}
      />

      {error && <p className="error">{error}</p>}

      {uploadedFilePath && originalFileName && (
        <div className="info-box">
          <p>Uploaded: {originalFileName} (Path: {uploadedFilePath})</p>
        </div>
      )}

      {isLoadingColumns && <div className="loader"></div>}

      {columnNames.length > 0 && !isLoadingColumns && (
        <>
          <ColumnSelector
            columnNames={columnNames}
            dateColumn={dateColumn}
            setDateColumn={setDateColumn}
            targetColumn={targetColumn}
            setTargetColumn={setTargetColumn}
            groupByColumns={groupByColumns}
            setGroupByColumns={setGroupByColumns}
            frequency={frequency}
            setFrequency={setFrequency}
          />
          <ModelSelector
            selectedModels={selectedModels}
            setSelectedModels={setSelectedModels}
            availableModels={['ARIMA', 'Prophet', 'ExponentialSmoothing', 'LSTM']}
          />
          <div className="component-section">
            <button onClick={handlePreprocessingSubmit} disabled={isProcessing || !dateColumn || !targetColumn}>
              {isProcessing && !preprocessingRunId ? 'Preprocessing...' : '2. Preprocess Data'}
            </button>
          </div>
        </>
      )}

      {preprocessingRunId && (
         <div className="component-section">
            <p className="success">Preprocessing Complete! Run ID: {preprocessingRunId}</p>
            <button onClick={handleRunForecasting} disabled={isProcessing}>
              {isProcessing ? 'Forecasting...' : '3. Run Forecasting'}
            </button>
        </div>
      )}

      {isProcessing && (!forecastResults && preprocessingRunId) && <div className="loader"></div>}

      {/* Results display will go here later */}
      {/* {forecastResults && (
        <div className="component-section">
          <h2>Forecast Results</h2>
          <p>Forecasting completed. Check console for detailed results.</p>
          <pre>{JSON.stringify(forecastResults, null, 2)}</pre>
        </div>
      )} */}

      {forecastResults && !isProcessing && (
        <ResultsDisplay forecastResults={forecastResults} />
      )}

    </div>
  );
}

export default App;
