import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getActiveApiUrl } from '../config/api';
import GradientOrb from '../components/GradientOrb';
import './VoiceChat.css';

interface AudioChatResponse {
    transcript: string;
    response_text: string[];
    response_audio_paths: string[];
    is_finished: boolean;
    depression_score?: number;
    consistency_status?: string;
}

type MicStatus = 'pending' | 'granted' | 'denied';
type ConversationState = 'idle' | 'bot_speaking' | 'recording' | 'processing';

// Audio detection thresholds
const VOICE_THRESHOLD = 0.05;     // Volume level to consider as "speaking"
const SILENCE_THRESHOLD = 0.02;   // Volume level to consider as "silence"
const SILENCE_DURATION_MS = 2000; // 2 seconds of silence after speaking
const MAX_RECORDING_MS = 15000;   // Maximum 15 seconds recording
const MAX_WAIT_FOR_SPEECH_MS = 8000; // If no speech after 8 seconds, stop

const VoiceChat: React.FC = () => {
    const navigate = useNavigate();
    const API_BASE = getActiveApiUrl();

    // Core state
    const [micStatus, setMicStatus] = useState<MicStatus>('pending');
    const [state, setState] = useState<ConversationState>('idle');
    const [turnCount, setTurnCount] = useState(0);


    // Display state
    const [botQuestion, setBotQuestion] = useState('');
    const [userResponse, setUserResponse] = useState('');

    // Session
    const [sessionId] = useState(() => `session_${Date.now()}`);

    // Refs for audio handling
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const silenceStartRef = useRef<number | null>(null);
    const hasSpokenRef = useRef<boolean>(false);
    const recordingStartRef = useRef<number | null>(null);  // Track when recording started
    const animationFrameRef = useRef<number | null>(null);
    const isCleanedUpRef = useRef<boolean>(false);

    // Cleanup function
    const cleanup = useCallback(() => {
        isCleanedUpRef.current = true;

        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            try {
                mediaRecorderRef.current.stop();
            } catch (e) { /* ignore */ }
            mediaRecorderRef.current = null;
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
    }, []);

    // Request mic permission on mount
    useEffect(() => {
        requestMicPermission();

        return () => {
            cleanup();
            const formData = new FormData();
            formData.append('session_id', sessionId);
            navigator.sendBeacon(`${API_BASE}/api/v1/audio/chat/cleanup`, formData);
        };
    }, []);

    const requestMicPermission = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            streamRef.current = stream;
            setMicStatus('granted');

            audioContextRef.current = new AudioContext();
            analyserRef.current = audioContextRef.current.createAnalyser();
            const source = audioContextRef.current.createMediaStreamSource(stream);
            source.connect(analyserRef.current);
            analyserRef.current.fftSize = 256;

            startConversation();
        } catch (error) {
            console.error('Mic permission denied:', error);
            setMicStatus('denied');
        }
    };

    const startConversation = async () => {
        if (isCleanedUpRef.current) return;
        setState('processing');

        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);

            const response = await fetch(`${API_BASE}/api/v1/audio/chat/start`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();
            const fullQuestion = data.response_text.join(' ');
            setBotQuestion(fullQuestion);

            if (data.response_audio_paths.length > 0) {
                await playAudioSequence(data.response_audio_paths);
            }

            if (!isCleanedUpRef.current) {
                startRecording();
            }
        } catch (error) {
            console.error('Error starting conversation:', error);
            setBotQuestion('Failed to start. Please check backend.');
        }
    };

    const playAudioSequence = async (paths: string[]): Promise<void> => {
        if (isCleanedUpRef.current) return;
        setState('bot_speaking');

        for (const path of paths) {
            if (isCleanedUpRef.current) return;
            await new Promise<void>((resolve) => {
                const audio = new Audio(`${API_BASE}${path}`);
                audio.onended = () => resolve();
                audio.onerror = () => resolve();
                audio.play().catch(() => resolve());
            });
        }
    };

    const startRecording = useCallback(() => {
        if (!streamRef.current || isCleanedUpRef.current) return;

        setState('recording');
        audioChunksRef.current = [];
        silenceStartRef.current = null;
        hasSpokenRef.current = false;
        recordingStartRef.current = Date.now();  // Track start time

        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
            ? 'audio/webm;codecs=opus'
            : MediaRecorder.isTypeSupported('audio/webm')
                ? 'audio/webm'
                : 'audio/mp4';

        const mediaRecorder = new MediaRecorder(streamRef.current, { mimeType });
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunksRef.current.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            if (isCleanedUpRef.current) return;
            const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
            audioChunksRef.current = [];
            await sendAudio(audioBlob);
        };

        mediaRecorder.start();
        detectSilence();
    }, []);

    const detectSilence = useCallback(() => {
        if (!analyserRef.current || state !== 'recording' || isCleanedUpRef.current) return;

        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255;
        const elapsed = Date.now() - (recordingStartRef.current || Date.now());

        // Safety: max recording duration
        if (elapsed >= MAX_RECORDING_MS) {
            console.log('[Voice] Max recording time reached');
            stopRecording();
            return;
        }

        // If user hasn't spoken after MAX_WAIT_FOR_SPEECH_MS, stop anyway
        if (!hasSpokenRef.current && elapsed >= MAX_WAIT_FOR_SPEECH_MS) {
            console.log('[Voice] No speech detected, stopping');
            stopRecording();
            return;
        }

        if (average >= VOICE_THRESHOLD) {
            hasSpokenRef.current = true;
            silenceStartRef.current = null;
        } else if (hasSpokenRef.current && average < SILENCE_THRESHOLD) {
            if (!silenceStartRef.current) {
                silenceStartRef.current = Date.now();
            } else if (Date.now() - silenceStartRef.current >= SILENCE_DURATION_MS) {
                stopRecording();
                return;
            }
        }

        animationFrameRef.current = requestAnimationFrame(detectSilence);
    }, [state]);

    useEffect(() => {
        if (state === 'recording' && !isCleanedUpRef.current) {
            detectSilence();
        }
        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [state, detectSilence]);

    const stopRecording = () => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.requestData();
            mediaRecorderRef.current.stop();
        }
    };

    const sendAudio = async (audioBlob: Blob) => {
        if (isCleanedUpRef.current) return;
        setState('processing');

        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);
            formData.append('audio_file', audioBlob, 'recording.webm');

            const response = await fetch(`${API_BASE}/api/v1/audio/chat/turn`, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Server error');
            }

            const data: AudioChatResponse = await response.json();

            if (isCleanedUpRef.current) return;

            setUserResponse(data.transcript);
            setTurnCount(prev => prev + 1);
            setBotQuestion(data.response_text.join(' '));

            if (data.is_finished) {

                const finalScore = data.depression_score;
                if (finalScore !== undefined) {

                }

                if (data.response_audio_paths.length > 0) {
                    await playAudioSequence(data.response_audio_paths);
                }
                setState('idle');

                const cleanupForm = new FormData();
                cleanupForm.append('session_id', sessionId);
                await fetch(`${API_BASE}/api/v1/audio/chat/cleanup`, {
                    method: 'POST',
                    body: cleanupForm,
                });

                // Wait 3 seconds then navigate to results
                setTimeout(() => {
                    navigate('/results', { state: { score: finalScore } });
                }, 3000);
            } else {
                if (data.response_audio_paths.length > 0) {
                    await playAudioSequence(data.response_audio_paths);
                }
                if (!isCleanedUpRef.current) {
                    startRecording();
                }
            }
        } catch (error) {
            console.error('Error sending audio:', error);
            setState('idle');
        }
    };

    const handleQuit = async () => {
        cleanup();
        setState('idle');

        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);
            await fetch(`${API_BASE}/api/v1/audio/chat/cleanup`, {
                method: 'POST',
                body: formData,
            });
        } catch (e) { /* ignore */ }

        navigate(-1); // Go back to previous page
    };

    const handleHome = async () => {
        cleanup();
        setState('idle');

        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);
            // Fire-and-forget cleanup
            fetch(`${API_BASE}/api/v1/audio/chat/cleanup`, {
                method: 'POST',
                body: formData,
            });
        } catch (e) { /* ignore */ }

        navigate('/', { replace: true });
    };

    // Mic denied overlay
    if (micStatus === 'denied') {
        return (
            <div className="voice-chat voice-chat--blocked">
                <div className="mic-denied">
                    <div className="mic-denied__icon">🎤</div>
                    <h2>Microphone Access Required</h2>
                    <p>Please allow microphone access to use voice chat.</p>
                    <button onClick={() => window.location.reload()}>Try Again</button>
                </div>
            </div>
        );
    }

    // Loading
    if (micStatus === 'pending') {
        return (
            <div className="voice-chat voice-chat--loading">
                <GradientOrb size="large" />
                <p className="voice-chat__loading-text">Requesting microphone access...</p>
            </div>
        );
    }

    return (
        <div className="voice-chat">
            <div className="voice-chat__header">
                <button className="voice-chat__back-orb" onClick={handleQuit} aria-label="Go back" title="Go back">
                    <GradientOrb size="small" />
                </button>
            </div>

            <div className="voice-chat__content">
                <div className={`voice-chat__orb-main ${state === 'bot_speaking' ? 'speaking' : ''} ${state === 'recording' ? 'listening' : ''} ${state === 'processing' ? 'processing' : ''}`}>
                    <GradientOrb size="large" />
                </div>

                <h2 key={botQuestion} className="voice-chat__question fade-in">{botQuestion || 'Starting conversation...'}</h2>

                {turnCount > 0 && userResponse && (
                    <p className="voice-chat__user-response">{userResponse}</p>
                )}

                <div className="voice-chat__status">
                    {state === 'bot_speaking' && <span className="status--speaking">Speaking...</span>}
                    {state === 'recording' && <span className="status--recording">Listening...</span>}
                    {state === 'processing' && <span className="status--processing">Processing...</span>}
                </div>

                <button className="voice-chat__quit" onClick={handleHome}>
                    Quit Chat
                </button>
            </div>
        </div>
    );
};

export default VoiceChat;
