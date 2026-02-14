/**
 * API Helper Hook
 * 
 * React hook to get the active API endpoint with auto-detection
 */

import { useState, useEffect } from 'react';
import { getApiUrl, API_BASE_URL } from '../config/api';

export function useApiUrl() {
    const [apiUrl, setApiUrl] = useState<string>(API_BASE_URL);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        getApiUrl().then((url) => {
            setApiUrl(url);
            setIsLoading(false);
        });
    }, []);

    return { apiUrl, isLoading };
}
