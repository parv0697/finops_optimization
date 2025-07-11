import React from 'react';

function ModelSelector({ selectedModels, setSelectedModels, availableModels }) {

  const handleModelChange = (event) => {
    const { value, checked } = event.target;
    if (checked) {
      setSelectedModels(prev => [...prev, value]);
    } else {
      setSelectedModels(prev => prev.filter(model => model !== value));
    }
  };

  return (
    <div className="component-section">
      <h2>Select Forecasting Models</h2>
      {availableModels.map(model => (
        <div key={model}>
          <input
            type="checkbox"
            id={`model-${model}`}
            value={model}
            checked={selectedModels.includes(model)}
            onChange={handleModelChange}
          />
          <label htmlFor={`model-${model}`}>{model}</label>
        </div>
      ))}
      {selectedModels.length === 0 && <p className="error">Please select at least one model.</p>}
    </div>
  );
}

export default ModelSelector;
