import React, { useState } from 'react';
import apiClient from '../services/apiService'; // Corrected path

function FileUpload({ onUploadSuccess, onUploadError }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setMessage('');
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setMessage('Please select a file first.');
      onUploadError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    setUploading(true);
    setMessage('Uploading...');

    try {
      const response = await apiClient.post('/upload_data', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setMessage(`Success: ${response.data.message}. Path: ${response.data.filepath}`);
      onUploadSuccess(response.data.filepath, selectedFile.name);
    } catch (error) {
      let errorMessage = 'Error uploading file.';
      if (error.response) {
        errorMessage = `Error: ${error.response.data.detail || error.response.statusText}`;
      } else if (error.request) {
        errorMessage = 'Error: No response from server. Check network or proxy.';
      } else {
        errorMessage = `Error: ${error.message}`;
      }
      setMessage(errorMessage);
      onUploadError(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="component-section">
      <h2>1. Upload FinOps Data</h2>
      <input type="file" accept=".csv" onChange={handleFileChange} disabled={uploading} />
      <button onClick={handleUpload} disabled={uploading || !selectedFile}>
        {uploading ? 'Uploading...' : 'Upload'}
      </button>
      {message && <p className={message.startsWith('Error') ? 'error' : 'success'}>{message}</p>}
    </div>
  );
}

export default FileUpload;
