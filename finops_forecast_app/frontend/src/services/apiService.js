// Placeholder for apiService
import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api/v1', // Proxied to backend (e.g., http://localhost:8000/api/v1)
  headers: {
    'Content-Type': 'application/json',
  },
});

export default apiClient;
