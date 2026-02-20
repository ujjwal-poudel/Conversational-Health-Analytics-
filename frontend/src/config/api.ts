/**
 * API Configuration with Auto-Failover
 * 
 * Automatically selects the best available API endpoint:
 * 1. Tries cloud endpoint first (Cloudflare HTTPS tunnel)
 * 2. Falls back to localhost if cloud is unavailable
 * 3. Logs the selected endpoint to console
 */

// Available endpoints (priority order)
// 1. Environment variable (for Production/Render) - MUST be set in Render dashboard
const ENV_API = import.meta.env.VITE_API_URL;
// 2. Localhost fallback (for Development) - configurable via env var
const LOCAL_API = import.meta.env.VITE_FALLBACK_API_URL;

// Cache key for localStorage
const CACHE_KEY = 'preferred_api_endpoint';
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

let selectedEndpoint: string | null = ENV_API || LOCAL_API;

/**
 * Test if an API endpoint is reachable
 */
async function testEndpoint(url: string): Promise<boolean> {
    if (!url) return false;
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000); // 3 second timeout

        const response = await fetch(`${url}/health`, {
            method: 'GET',
            signal: controller.signal,
        });

        clearTimeout(timeoutId);
        return response.ok;
    } catch (error) {
        return false;
    }
}

/**
 * Get cached endpoint if still valid
 */
function getCachedEndpoint(): string | null {
    try {
        const cached = localStorage.getItem(CACHE_KEY);
        if (!cached) return null;

        const { endpoint, timestamp } = JSON.parse(cached);
        const age = Date.now() - timestamp;

        // Cache valid for 5 minutes
        if (age < CACHE_TTL) {
            return endpoint;
        }

        // Cache expired
        localStorage.removeItem(CACHE_KEY);
        return null;
    } catch {
        return null;
    }
}

/**
 * Cache the selected endpoint
 */
function cacheEndpoint(endpoint: string): void {
    try {
        localStorage.setItem(CACHE_KEY, JSON.stringify({
            endpoint,
            timestamp: Date.now()
        }));
    } catch {
        // Ignore localStorage errors
    }
}

/**
 * Select the best available API endpoint
 * Tries env var first, falls back to local
 */
async function selectBestEndpoint(checkHealth: boolean = true): Promise<string> {
    // Check cache first
    const cached = getCachedEndpoint();
    if (cached) {
        console.log('📦 [API Config] Using cached endpoint:', cached);
        selectedEndpoint = cached;

        // If not checking health, return cached immediately
        if (!checkHealth) {
            return cached;
        }

        // Verify cached endpoint still works
        const stillWorks = await testEndpoint(cached);
        if (stillWorks) {
            console.log('✅ [API Config] Cached endpoint is healthy!');
            return cached;
        }
        console.log('❌ [API Config] Cached endpoint no longer reachable, retesting...');
        localStorage.removeItem(CACHE_KEY);
    }

    // If not checking health, return Env or Local default immediately
    if (!checkHealth) {
        const defaultApi = ENV_API || LOCAL_API;
        console.log(`🔍 [API Config] Using default endpoint: ${defaultApi} (health check deferred)`);
        selectedEndpoint = defaultApi;
        return defaultApi;
    }

    console.log('🔍 [API Config] Testing endpoints...');

    // 1. Try Environment Variable (Production)
    if (ENV_API) {
        console.log(`   Trying configured API: ${ENV_API}`);
        const envWorks = await testEndpoint(ENV_API);
        if (envWorks) {
            console.log('✅ [API Config] Using Configured API:', ENV_API);
            selectedEndpoint = ENV_API;
            cacheEndpoint(ENV_API);
            return ENV_API;
        }
        console.log('❌ [API Config] Configured API not reachable');
    }

    // 2. Try Localhost (Development)
    console.log(`   Trying local: ${LOCAL_API}`);
    const localWorks = await testEndpoint(LOCAL_API);
    if (localWorks) {
        console.log('✅ [API Config] Using LOCAL endpoint:', LOCAL_API);
        selectedEndpoint = LOCAL_API;
        cacheEndpoint(LOCAL_API);
        return LOCAL_API;
    }

    // Fallback: Default to Env if set, else Local
    const fallback = ENV_API || LOCAL_API;
    console.error(`⚠️  [API Config] No backend reachable! Defaulting to ${fallback}...`);
    selectedEndpoint = fallback;
    return fallback;
}

// Initialize endpoint selection
let endpointPromise: Promise<string> | null = null;

/**
 * Get the active API base URL (async, lazy health check on first call)
 * Returns cached/default immediately, verifies health in background
 */
export async function getApiUrl(): Promise<string> {
    if (selectedEndpoint) {
        return selectedEndpoint; // Already selected
    }

    if (!endpointPromise) {
        // First call: check cache and verify it works
        const cached = getCachedEndpoint();
        if (cached) {
            console.log('📦 [API Config] Found cached endpoint, testing...');
            const works = await testEndpoint(cached);
            if (works) {
                console.log('✅ [API Config] Cached endpoint is healthy:', cached);
                selectedEndpoint = cached;
                return cached;
            }
            console.log('❌ [API Config] Cached endpoint failed, clearing cache');
            localStorage.removeItem(CACHE_KEY);
        }

        // No valid cache: test endpoints and select best
        console.log('🔍 [API Config] Selecting endpoint...');
        // Disable health check if ENV_API is provided to avoid cold-start timeouts
        const checkHealth = !ENV_API;
        const endpoint = await selectBestEndpoint(checkHealth);
        selectedEndpoint = endpoint;
        (globalThis as any).__ACTIVE_API_URL__ = endpoint;
        return endpoint;
    }

    const endpoint = await endpointPromise;

    // Update the exported constant for synchronous usage
    (globalThis as any).__ACTIVE_API_URL__ = endpoint;

    return endpoint;
}

/**
 * Synchronous API URL getter
 * Returns the detected endpoint after initialization, or cloud default before
 */
export function getActiveApiUrl(): string {
    // Prefer cloud with auto-failover to local
    return selectedEndpoint || LOCAL_API;
}

export const API_BASE_URL = ENV_API || LOCAL_API;

/**
 * Force re-check endpoints (useful for error recovery)
 */
export function resetEndpointSelection() {
    console.log('🔄 [API Config] Resetting endpoint selection and cache...');
    selectedEndpoint = null;
    endpointPromise = null;
    localStorage.removeItem(CACHE_KEY);
}
