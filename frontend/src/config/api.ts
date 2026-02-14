/**
 * API Configuration with Auto-Failover
 * 
 * Automatically selects the best available API endpoint:
 * 1. Tries cloud endpoint first (Cloudflare HTTPS tunnel)
 * 2. Falls back to localhost if cloud is unavailable
 * 3. Logs the selected endpoint to console
 */

// Available endpoints (priority order)
const CLOUD_API = 'https://penalty-publicity-vintage-gamecube.trycloudflare.com';
const LOCAL_API = 'http://localhost:8000';

// Cache key for localStorage
const CACHE_KEY = 'preferred_api_endpoint';
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

let selectedEndpoint: string | null = null;

/**
 * Test if an API endpoint is reachable
 */
async function testEndpoint(url: string): Promise<boolean> {
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
 * Tries cloud first, falls back to local
 * Returns immediately with cached/default if checkHealth=false
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
    
    // If not checking health, return cloud default immediately
    if (!checkHealth) {
        console.log('🔍 [API Config] Using CLOUD endpoint (default, health check deferred)');
        selectedEndpoint = CLOUD_API;
        return CLOUD_API;
    }
    
    console.log('🔍 [API Config] Testing endpoints...');
    
    // Try cloud first (priority)
    console.log(`   Trying cloud: ${CLOUD_API}`);
    const cloudWorks = await testEndpoint(CLOUD_API);
    
    if (cloudWorks) {
        console.log('✅ [API Config] Using CLOUD endpoint:', CLOUD_API);
        console.log('   Cloud backend is online and healthy!');
        selectedEndpoint = CLOUD_API;
        cacheEndpoint(CLOUD_API);
        return CLOUD_API;
    }
    
    console.log('❌ [API Config] Cloud endpoint not reachable');
    
    // Fallback to local
    console.log(`   Trying local: ${LOCAL_API}`);
    const localWorks = await testEndpoint(LOCAL_API);
    
    if (localWorks) {
        console.log('✅ [API Config] Using LOCAL endpoint:', LOCAL_API);
        console.log('   Running with local backend');
        selectedEndpoint = LOCAL_API;
        cacheEndpoint(LOCAL_API);
        return LOCAL_API;
    }
    
    // Neither works - default to cloud and let requests fail with proper errors
    console.error('⚠️  [API Config] No backend available! Defaulting to cloud...');
    console.error('   Please ensure backend is running (cloud or local)');
    selectedEndpoint = CLOUD_API;
    // Don't cache when nothing works
    return CLOUD_API;
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
        console.log('🔍 [API Config] Testing endpoints...');
        const endpoint = await selectBestEndpoint(true);
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
    return selectedEndpoint || CLOUD_API;
}

/**
 * Legacy export for compatibility (will be updated after initialization)
 */
export const API_BASE_URL = CLOUD_API;

/**
 * Force re-check endpoints (useful for error recovery)
 */
export function resetEndpointSelection() {
    console.log('🔄 [API Config] Resetting endpoint selection and cache...');
    selectedEndpoint = null;
    endpointPromise = null;
    localStorage.removeItem(CACHE_KEY);
}
