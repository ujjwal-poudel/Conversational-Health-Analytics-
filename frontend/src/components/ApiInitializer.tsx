/**
 * API Initializer Component
 * 
 * Initializes API endpoint detection on app startup
 * Shows a loading screen while testing endpoints
 */

import { useEffect, useState } from 'react';
import { getApiUrl } from '../config/api';

interface ApiInitializerProps {
    children: React.ReactNode;
}

export function ApiInitializer({ children }: ApiInitializerProps) {
    const [isReady, setIsReady] = useState(false);
    const [showLoading, setShowLoading] = useState(false);

    useEffect(() => {
        // Delay showing loading screen by 200ms
        // If endpoint test completes quickly (cached), user won't see loading
        const loadingTimer = setTimeout(() => setShowLoading(true), 200);
        
        // Test and select best endpoint on app startup
        getApiUrl().then(() => {
            clearTimeout(loadingTimer);
            setIsReady(true);
        });
        
        return () => clearTimeout(loadingTimer);
    }, []);

    if (!isReady && !showLoading) {
        // Silent loading for first 200ms (cached endpoint check)
        return null;
    }

    if (!isReady) {
        return (
            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100vh',
                fontFamily: 'system-ui, -apple-system, sans-serif',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white'
            }}>
                <div style={{
                    fontSize: '24px',
                    marginBottom: '20px',
                    animation: 'pulse 2s infinite'
                }}>
                    🔗 Connecting to backend...
                </div>
                <div style={{
                    fontSize: '14px',
                    opacity: 0.8
                }}>
                    Testing cloud and local endpoints
                </div>
                <style>{`
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                `}</style>
            </div>
        );
    }

    return <>{children}</>;
}
