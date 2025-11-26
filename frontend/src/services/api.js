import axios from 'axios';

// Determine API base URL
// Use proxy on Vercel to avoid mixed content (HTTPS -> HTTP) issues
// On localhost, connect directly
const isVercel = typeof window !== 'undefined' && window.location.hostname.includes('vercel.app');
const useProxy = isVercel || process.env.REACT_APP_USE_PROXY === 'true';

const API_BASE_URL = useProxy
  ? '/api' // Use proxy on Vercel
  : (process.env.REACT_APP_API_URL || 'http://54.198.232.153:8000');

// Log the API URL being used (helpful for debugging)
console.log('API Base URL:', API_BASE_URL);
console.log('Environment:', process.env.NODE_ENV);
console.log('Using Proxy:', useProxy);
console.log('Is Vercel:', isVercel);

const api = axios.create({
  baseURL: API_BASE_URL,
  // Don't set default Content-Type - let axios set it based on data type
  // For FormData, axios will set multipart/form-data with boundary
  // For JSON, axios will set application/json
  timeout: 10000, // 10 second timeout for most requests
});

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] Request: ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
    console.log(`[API] Full URL: ${config.baseURL}${config.url}`);
    console.log(`[API] Headers:`, config.headers);
    if (config.data instanceof FormData) {
      console.log(`[API] Body is FormData`);
      // Log FormData entries if possible
      for (let pair of config.data.entries()) {
        console.log(`[API] FormData: ${pair[0]} = ${pair[1] instanceof File ? `File(${pair[1].name}, ${pair[1].size} bytes)` : pair[1]}`);
      }
    }
    return config;
  },
  (error) => {
    console.error('[API] Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Response: ${response.status} ${response.config.url}`, response.data);
    return response;
  },
  (error) => {
    // Enhanced error logging
    const errorDetails = {
      message: error.message,
      status: error.response?.status,
      statusText: error.response?.statusText,
      data: error.response?.data,
      url: error.config?.url,
      baseURL: error.config?.baseURL,
      fullURL: error.config?.baseURL + error.config?.url,
      code: error.code, // Network errors have codes like 'ERR_NETWORK', 'ERR_CORS', etc.
      request: error.request
    };

    console.error('API Response Error:', errorDetails);

    // Provide more helpful error messages
    if (error.code === 'ERR_NETWORK' || error.code === 'ERR_INTERNET_DISCONNECTED') {
      error.userMessage = 'Network error: Unable to connect to the backend server. Please check if the server is running.';
    } else if (error.code === 'ERR_CORS') {
      error.userMessage = 'CORS error: The backend server is not allowing requests from this origin. Please check CORS configuration.';
    } else if (error.message?.includes('Mixed Content')) {
      error.userMessage = 'Mixed Content error: Cannot make HTTP requests from HTTPS page. The backend needs to use HTTPS.';
    } else if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
      error.userMessage = 'Request timeout: The server took too long to respond. Please try again.';
    } else if (error.response) {
      error.userMessage = error.response.data?.detail || error.response.data?.message || `Server error: ${error.response.status} ${error.response.statusText}`;
    } else {
      error.userMessage = error.message || 'An unexpected error occurred. Please check the console for details.';
    }

    return Promise.reject(error);
  }
);

export const uploadPDF = async (file, startPage, endPage, generateSummary = false, voiceProvider = 'openai', openaiVoice = null, cartesiaVoiceId = null, cartesiaModelId = null) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('start_page', startPage);
  formData.append('end_page', endPage);
  formData.append('generate_summary', generateSummary);
  formData.append('voice_provider', voiceProvider);
  if (voiceProvider === 'openai' && openaiVoice) {
    formData.append('openai_voice', openaiVoice);
  }
  if (cartesiaVoiceId) {
    formData.append('cartesia_voice_id', cartesiaVoiceId);
  }
  if (cartesiaModelId) {
    formData.append('cartesia_model_id', cartesiaModelId);
  }

  // Don't set Content-Type - axios will set it automatically with boundary for FormData
  // baseURL is already '/api', so just use '/upload' not '/api/upload'
  const response = await api.post('/upload', formData, {
    timeout: 60000, // 60 seconds for file upload
  });
  return response.data;
};

export const getJobStatus = async (jobId) => {
  // baseURL is already '/api', so just use '/jobs/...' not '/api/jobs/...'
  // URL-encode the jobId to handle spaces and special characters
  const encodedJobId = encodeURIComponent(jobId);
  const response = await api.get(`/jobs/${encodedJobId}`, {
    timeout: 5000, // 5 seconds for status checks (should be fast)
  });
  return response.data;
};

export const downloadVideo = async (jobId) => {
  // URL-encode the jobId to handle spaces and special characters
  const encodedJobId = encodeURIComponent(jobId);
  const response = await api.get(`/jobs/${encodedJobId}/download/video`, {
    responseType: 'blob',
  });
  return response.data;
};

export const downloadSummary = async (jobId) => {
  // URL-encode the jobId to handle spaces and special characters
  const encodedJobId = encodeURIComponent(jobId);
  const response = await api.get(`/jobs/${encodedJobId}/download/summary`, {
    responseType: 'text',
  });
  return response.data;
};

export const generateSummaryVideo = async (jobId) => {
  // URL-encode the jobId to handle spaces and special characters
  const encodedJobId = encodeURIComponent(jobId);
  const response = await api.post(`/jobs/${encodedJobId}/generate-summary-video`);
  return response.data;
};

export const downloadSummaryVideo = async (jobId) => {
  // URL-encode the jobId to handle spaces and special characters
  const encodedJobId = encodeURIComponent(jobId);
  const response = await api.get(`/jobs/${encodedJobId}/download/summary-video`, {
    responseType: 'blob',
  });
  return response.data;
};

export const generateSummary = async (jobId) => {
  // URL-encode the jobId to handle spaces and special characters
  const encodedJobId = encodeURIComponent(jobId);
  const response = await api.post(`/jobs/${encodedJobId}/generate-summary`);
  return response.data;
};

export const summarizePDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  // Don't set Content-Type - axios will set it automatically with boundary for FormData
  // baseURL is already '/api', so just use '/summarize-pdf' not '/api/summarize-pdf'
  const response = await api.post('/summarize-pdf', formData, {
    timeout: 300000, // 5 minutes for summarization
  });
  return response.data;
};

export const generateVideoFromText = async (text, voiceProvider = 'openai', openaiVoice = null, cartesiaVoiceId = null, cartesiaModelId = null) => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('voice_provider', voiceProvider);
  if (voiceProvider === 'openai' && openaiVoice) {
    formData.append('openai_voice', openaiVoice);
  }
  if (cartesiaVoiceId) {
    formData.append('cartesia_voice_id', cartesiaVoiceId);
  }
  if (cartesiaModelId) {
    formData.append('cartesia_model_id', cartesiaModelId);
  }

  // Don't set Content-Type - axios will set it automatically with boundary for FormData
  // baseURL is already '/api', so just use '/generate-video-from-text' not '/api/generate-video-from-text'
  const response = await api.post('/generate-video-from-text', formData, {
    timeout: 600000, // 10 minutes for video generation
  });
  return response.data;
};

export const generateReelsVideo = async (text, voiceProvider = 'openai', openaiVoice = null, cartesiaVoiceId = null, cartesiaModelId = null) => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('voice_provider', voiceProvider);
  if (voiceProvider === 'openai' && openaiVoice) {
    formData.append('openai_voice', openaiVoice);
  }
  if (cartesiaVoiceId) {
    formData.append('cartesia_voice_id', cartesiaVoiceId);
  }
  if (cartesiaModelId) {
    formData.append('cartesia_model_id', cartesiaModelId);
  }

  // Don't set Content-Type - axios will set it automatically with boundary for FormData
  // baseURL is already '/api', so just use '/generate-reels-video' not '/api/generate-reels-video'
  const response = await api.post('/generate-reels-video', formData, {
    timeout: 600000, // 10 minutes for video generation
  });
  return response.data;
};

// Cartesia API functions
export const listCartesiaVoices = async (language = null, tags = null) => {
  const params = {};
  if (language) params.language = language;
  if (tags) params.tags = tags;
  // baseURL is already '/api', so just use '/cartesia/voices' not '/api/cartesia/voices'
  const response = await api.get('/cartesia/voices', { params });
  return response.data;
};

export const getCartesiaVoice = async (voiceId) => {
  const response = await api.get(`/cartesia/voices/${voiceId}`);
  return response.data;
};

export const listCartesiaModels = async () => {
  const response = await api.get('/cartesia/models');
  return response.data;
};

export default api;

