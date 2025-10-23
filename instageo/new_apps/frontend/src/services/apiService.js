import { INSTAGEO_BACKEND_API_ENDPOINTS } from '../config';
import { logger } from '../utils/logger';

class ApiService {
  async get(endpoint, getAccessTokenSilently) {
    return this.makeRequest(endpoint, { method: 'GET' }, getAccessTokenSilently);
  }

  async fetchImageDataFromURL(url, getAccessTokenSilently) {
    const authHeaders = await this.getAuthHeaders(getAccessTokenSilently);
    const headers = {
      ...authHeaders,
    };

    const response = await fetch(url, { headers });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response;
  }

  async getAuthHeaders(getAccessTokenSilently) {
    try {
      const token = await getAccessTokenSilently();
      return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      };
    } catch (error) {
      logger.error('Failed to get access token:', error);
      return {
        'Content-Type': 'application/json'
      };
    }
  }

  async makeRequest(endpoint, options = {}, getAccessTokenSilently) {
    const authHeaders = await this.getAuthHeaders(getAccessTokenSilently);
    const config = {
      ...options,
      headers: {
        ...authHeaders,
        ...options.headers
      }
    };

    const response = await fetch(endpoint, config);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return response.json();
  }

  async runModel(payload, getAccessTokenSilently) {
    return this.makeRequest(
      INSTAGEO_BACKEND_API_ENDPOINTS.RUN_MODEL,
      {
        method: 'POST',
        body: JSON.stringify(payload)
      },
      getAccessTokenSilently
    );
  }

  async getTaskStatus(taskId, getAccessTokenSilently) {
    return this.makeRequest(
      INSTAGEO_BACKEND_API_ENDPOINTS.TASK_STATUS(taskId),
      { method: 'GET' },
      getAccessTokenSilently
    );
  }

  async getAllTasks(getAccessTokenSilently) {
    return this.makeRequest(
      INSTAGEO_BACKEND_API_ENDPOINTS.GET_ALL_TASKS,
      { method: 'GET' },
      getAccessTokenSilently
    );
  }

  async getModels(getAccessTokenSilently) {
    return this.makeRequest(
      INSTAGEO_BACKEND_API_ENDPOINTS.GET_MODELS,
      { method: 'GET' },
      getAccessTokenSilently
    );
  }

  async visualizeTask(taskId, getAccessTokenSilently) {
    return this.makeRequest(
      INSTAGEO_BACKEND_API_ENDPOINTS.VISUALIZE(taskId),
      { method: 'GET' },
      getAccessTokenSilently
    );
  }
}

const apiService = new ApiService();
export default apiService;
