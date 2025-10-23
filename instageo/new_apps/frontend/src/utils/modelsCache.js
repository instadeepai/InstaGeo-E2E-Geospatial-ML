import apiService from '../services/apiService';

const MODELS_CACHE_KEY = 'instageo_models_cache_v2';
const MODELS_TTL_MS = 24 * 60 * 60 * 1000; // 24h

export async function fetchModelsWithTTL(getAccessTokenSilently) {
  const now = Date.now();

  try {
    const cachedRaw = localStorage.getItem(MODELS_CACHE_KEY);
    if (cachedRaw) {
      const cached = JSON.parse(cachedRaw);
      if (now - cached.timestamp < MODELS_TTL_MS) {
        return cached.data;
      }
      localStorage.removeItem(MODELS_CACHE_KEY);
    }
  } catch {
    localStorage.removeItem(MODELS_CACHE_KEY);
  }

  const data = await apiService.getModels(getAccessTokenSilently);

  try {
    localStorage.setItem(
      MODELS_CACHE_KEY,
      JSON.stringify({ timestamp: now, data })
    );
  } catch {
    // ignore storage errors
  }

  return data;
}

export function clearModelsCache() {
  try { localStorage.removeItem(MODELS_CACHE_KEY); } catch {}
}
