import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function ResultsDisplay({ forecastResults }) {
  if (!forecastResults || !forecastResults.results_by_group) {
    return <p>No forecast results to display yet. Run forecasting first.</p>;
  }

  if (forecastResults.global_errors && forecastResults.global_errors.length > 0) {
    return (
      <div className="component-section error">
        <h3>Global Errors During Forecasting:</h3>
        {forecastResults.global_errors.map((err, idx) => <p key={idx}>{err}</p>)}
      </div>
    );
  }

  // Helper to transform forecast/actual data for charts
  const formatDataForChart = (modelPerformance, fullSeriesData = null) => {
    const chartData = [];
    let allDates = new Set();

    // Add actual test values
    if (modelPerformance.actual_test_values) {
      Object.keys(modelPerformance.actual_test_values).forEach(dateStr => allDates.add(dateStr));
    }
    // Add forecast values
    if (modelPerformance.forecast_values) {
      Object.keys(modelPerformance.forecast_values).forEach(dateStr => allDates.add(dateStr));
    }
    // Add full series data if provided (for historical actuals)
    // This part is more complex as it requires fetching/passing the preprocessed series.
    // For now, we'll focus on test actuals and forecasts.
    // if (fullSeriesData) {
    //   fullSeriesData.forEach(dp => allDates.add(dp.date)); // Assuming fullSeriesData is [{date: 'YYYY-MM-DD', value: Number}]
    // }


    const sortedDates = Array.from(allDates).sort((a, b) => new Date(a) - new Date(b));

    sortedDates.forEach(dateStr => {
      let dataPoint = { date: dateStr };
      // if (fullSeriesData) {
      //   const actualHist = fullSeriesData.find(dp => dp.date === dateStr);
      //   if (actualHist) dataPoint.actual_historical = actualHist.value;
      // }
      if (modelPerformance.actual_test_values && modelPerformance.actual_test_values[dateStr] !== undefined) {
        dataPoint.actual_test = modelPerformance.actual_test_values[dateStr];
      }
      if (modelPerformance.forecast_values && modelPerformance.forecast_values[dateStr] !== undefined) {
        dataPoint.forecast = modelPerformance.forecast_values[dateStr];
      }
      chartData.push(dataPoint);
    });
    return chartData;
  };


  return (
    <div className="component-section">
      <h2>4. Forecast Results</h2>
      {forecastResults.results_by_group.map((groupResult, index) => (
        <div key={groupResult.group_identifier || index} className="group-result-section container">
          <h3>Group: {groupResult.group_identifier}</h3>

          {groupResult.error_message && <p className="error">Error for this group: {groupResult.error_message}</p>}

          <p><strong>Best Model (by MAPE):</strong> {groupResult.best_model_by_mape || 'N/A'}</p>
          <p><strong>Best Model (by RMSE):</strong> {groupResult.best_model_by_rmse || 'N/A'}</p>

          <h4>Model Performance Metrics:</h4>
          <table border="1" style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr>
                <th>Model</th>
                <th>MAE</th>
                <th>MSE</th>
                <th>RMSE</th>
                <th>MAPE (%)</th>
                <th>sMAPE (%)</th>
              </tr>
            </thead>
            <tbody>
              {groupResult.model_performances.map((perf, pIndex) => (
                <tr key={pIndex}>
                  <td>{perf.model_name}</td>
                  <td>{perf.metrics.mae?.toFixed(3) ?? 'N/A'}</td>
                  <td>{perf.metrics.mse?.toFixed(3) ?? 'N/A'}</td>
                  <td>{perf.metrics.rmse?.toFixed(3) ?? 'N/A'}</td>
                  <td>{perf.metrics.mape?.toFixed(2) ?? 'N/A'}</td>
                  <td>{perf.metrics.smape?.toFixed(2) ?? 'N/A'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h4>Forecast Charts:</h4>
          {groupResult.model_performances.map((perf, pIndex) => {
            // Only render chart if there are forecast values
            if (!perf.forecast_values || Object.keys(perf.forecast_values).length === 0) {
              return <p key={`no-chart-${pIndex}`}>No forecast data to plot for {perf.model_name}. Summary: {perf.train_summary}</p>;
            }
            const chartData = formatDataForChart(perf);
            if (chartData.length === 0) {
                 return <p key={`no-chart-data-${pIndex}`}>Not enough data points to plot chart for {perf.model_name}.</p>;
            }

            return (
              <div key={`chart-container-${pIndex}`} style={{ marginTop: '20px' }}>
                <h5>Chart for Model: {perf.model_name}</h5>
                {perf.train_summary && perf.train_summary.startsWith("Failed:") && <p className="error">{perf.train_summary}</p>}
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    {/* Line for Actual Historical Data (if available) */}
                    {/* <Line type="monotone" dataKey="actual_historical" stroke="#8884d8" name="Actual (Historical)" dot={false}/> */}
                    {/* Line for Actual Test Data */}
                    {chartData.some(d => d.actual_test !== undefined) &&
                        <Line type="monotone" dataKey="actual_test" stroke="#82ca9d" name="Actual (Test)" dot={false} />
                    }
                    {/* Line for Forecast Data */}
                    <Line type="monotone" dataKey="forecast" stroke="#ff7300" name="Forecast" dot={false}/>
                  </LineChart>
                </ResponsiveContainer>
                {perf.train_summary && !perf.train_summary.startsWith("Failed:") && (
                    <details style={{marginTop: '10px'}}>
                        <summary>View {perf.model_name} Training Summary</summary>
                        <pre style={{whiteSpace: 'pre-wrap', backgroundColor: '#f0f0f0', padding: '10px', borderRadius: '4px', maxHeight: '200px', overflowY: 'auto'}}>
                            {perf.train_summary}
                        </pre>
                    </details>
                )}
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

export default ResultsDisplay;
