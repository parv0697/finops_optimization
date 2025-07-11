import React from 'react';

function ColumnSelector({
  columnNames,
  dateColumn, setDateColumn,
  targetColumn, setTargetColumn,
  groupByColumns, setGroupByColumns,
  frequency, setFrequency
}) {

  const handleGroupByChange = (event) => {
    const { options } = event.target;
    const selectedValues = [];
    for (let i = 0, l = options.length; i < l; i += 1) {
      if (options[i].selected) {
        selectedValues.push(options[i].value);
      }
    }
    setGroupByColumns(selectedValues);
  };

  return (
    <div className="component-section">
      <h2>Configure Data Columns & Frequency</h2>
      <div>
        <label htmlFor="dateColumn">Date Column:</label>
        <select id="dateColumn" value={dateColumn} onChange={(e) => setDateColumn(e.target.value)}>
          <option value="">Select Date Column</option>
          {columnNames.map(col => <option key={col} value={col}>{col}</option>)}
        </select>
      </div>
      <div>
        <label htmlFor="targetColumn">Target Column (Cost):</label>
        <select id="targetColumn" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
          <option value="">Select Target Column</option>
          {columnNames.map(col => <option key={col} value={col}>{col}</option>)}
        </select>
      </div>
      <div>
        <label htmlFor="groupByColumns">Group By Columns (Optional):</label>
        <select id="groupByColumns" multiple value={groupByColumns} onChange={handleGroupByChange} size="5">
          {/* <option value="">None</option> */}
          {columnNames.filter(col => col !== dateColumn && col !== targetColumn).map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
        </select>
         <small style={{display: 'block', marginLeft: '5px'}}> (Ctrl/Cmd + click to select multiple)</small>
      </div>
      <div>
        <label htmlFor="frequency">Frequency:</label>
        <select id="frequency" value={frequency} onChange={(e) => setFrequency(e.target.value)}>
          <option value="D">Daily</option>
          <option value="W">Weekly</option>
          <option value="M">Monthly</option>
        </select>
      </div>
    </div>
  );
}

export default ColumnSelector;
