import { useState, useRef, useEffect } from 'react';
import { getActiveApiUrl } from '../config/api';
import './AudioChat.css';

interface Message {
    text: string;
    isBot: boolean;
    audioPath?: string;
}

interface AudioChatResponse {
    transcript: string;
    timestamps?: any;
    user_audio_path: string;
    response_text: string[];
    response_audio_paths: string[];
    is_finished: boolean;
    depression_score?: number;
    merged_audio?: {
        user_only_path: string;
        full_conversation_path: string;
    };
}

const AudioChat = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [sessionId] = useState(() => `session_${Date.now()}`);
    const [conversationStarted, setConversationStarted] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [score, setScore] = useState<number | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const API_BASE_URL = getActiveApiUrl();

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Cleanup on page refresh/close if conversation is active but not finished
    useEffect(() => {
        const handleBeforeUnload = () => {
            if (conversationStarted && !isFinished) {
                const formData = new FormData();
                formData.append('session_id', sessionId);
                // sendBeacon is more reliable than fetch during page unload
                navigator.sendBeacon(`${API_BASE_URL}/api/v1/audio/chat/cleanup`, formData);
            }
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    }, [conversationStarted, isFinished, sessionId]);

    const startConversation = async () => {
        setIsProcessing(true);
        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);

            const response = await fetch(`${API_BASE_URL}/api/v1/audio/chat/start`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            // Add bot messages
            const botMessages: Message[] = data.response_text.map((text: string, idx: number) => ({
                text,
                isBot: true,
                audioPath: data.response_audio_paths[idx]
            }));

            setMessages(botMessages);
            setConversationStarted(true);

            // Auto-play audio sequence
            if (data.response_audio_paths.length > 0) {
                playAudioSequence(data.response_audio_paths);
            }
        } catch (error) {
            console.error('Error starting conversation:', error);
            alert('Failed to start conversation. Make sure backend is running.');
        } finally {
            setIsProcessing(false);
        }
    };

    const [isPlaying, setIsPlaying] = useState(false);

    const playAudioSequence = async (paths: string[]) => {
        setIsPlaying(true);
        for (const path of paths) {
            await new Promise<void>((resolve) => {
                // Backend returns relative path /audio/..., so we prepend base URL
                const audio = new Audio(`${API_BASE_URL}${path}`);
                audio.onended = () => resolve();
                audio.onerror = (e) => {
                    console.error("Audio playback error:", e);
                    resolve();
                };
                audio.play().catch(e => {
                    console.error("Audio play failed:", e);
                    resolve();
                });
            });
        }
        setIsPlaying(false);
    };

    const playAudio = (audioPath: string) => {
        playAudioSequence([audioPath]);
    };

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = async () => {
                // Browser will use default format (usually WebM)
                // Backend will auto-convert to WAV
                const audioBlob = new Blob(audioChunksRef.current);
                await sendAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error('Error accessing microphone:', error);
            alert('Failed to access microphone. Please grant permission.');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const cleanupSession = async () => {
        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);
            await fetch(`${API_BASE_URL}/api/v1/audio/chat/cleanup`, {
                method: 'POST',
                body: formData,
            });
            console.log("Session cleanup requested");
        } catch (error) {
            console.error("Cleanup failed:", error);
        }
    };

    const sendAudio = async (audioBlob: Blob) => {
        setIsProcessing(true);

        try {
            const formData = new FormData();
            formData.append('session_id', sessionId);
            formData.append('audio_file', audioBlob, 'recording.wav');

            const response = await fetch(`${API_BASE_URL}/api/v1/audio/chat/turn`, {
                method: 'POST',
                body: formData,
            });

            const data: AudioChatResponse = await response.json();

            // Add user message (transcript)
            const userMessage: Message = {
                text: data.transcript,
                isBot: false,
            };

            // Add bot responses
            const botMessages: Message[] = data.response_text.map((text: string, idx: number) => ({
                text,
                isBot: true,
                audioPath: data.response_audio_paths[idx]
            }));

            setMessages(prev => [...prev, userMessage, ...botMessages]);

            // Play bot response audio sequence
            if (data.response_audio_paths.length > 0) {
                await playAudioSequence(data.response_audio_paths);
            }

            // Check if finished
            if (data.is_finished) {
                setIsFinished(true);
                if (data.depression_score !== undefined) {
                    setScore(data.depression_score);
                }
                // Trigger cleanup after playback is done
                await cleanupSession();
            }
        } catch (error) {
            console.error('Error sending audio:', error);
            alert('Failed to process audio. Please try again.');
        } finally {
            setIsProcessing(false);
        }
    };

    const resetConversation = () => {
        setMessages([]);
        setConversationStarted(false);
        setIsFinished(false);
        setScore(null);
        window.location.reload(); // Reload to get new session ID
    };

    return (
        <div className="audio-chat-container">
            <div className="audio-chat-header">
                <h2>🎤 Voice Conversation</h2>
                <p>Have a spoken conversation with the AI therapist</p>
            </div>

            {!conversationStarted ? (
                <div className="start-section">
                    <button
                        onClick={startConversation}
                        disabled={isProcessing}
                        className="start-button"
                    >
                        {isProcessing ? '⏳ Starting...' : '▶️ Start Voice Conversation'}
                    </button>
                </div>
            ) : (
                <>
                    <div className="messages-container">
                        {messages.map((msg, idx) => (
                            <div key={idx} className={`message ${msg.isBot ? 'bot-message' : 'user-message'}`}>
                                <div className="message-content">
                                    <div className="message-icon">{msg.isBot ? '🤖' : '👤'}</div>
                                    <div className="message-text">{msg.text}</div>
                                    {msg.audioPath && (
                                        <button
                                            className="play-button"
                                            onClick={() => playAudio(msg.audioPath!)}
                                            title="Play audio"
                                        >
                                            🔊
                                        </button>
                                    )}
                                </div>
                            </div>
                        ))}
                        {isProcessing && (
                            <div className="message bot-message">
                                <div className="message-content">
                                    <div className="message-icon">🤖</div>
                                    <div className="message-text typing-indicator">
                                        <span></span><span></span><span></span>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    <div className="controls-container">
                        {!isFinished ? (
                            <button
                                className={`record-button ${isRecording ? 'recording' : ''}`}
                                onClick={isRecording ? stopRecording : startRecording}
                                disabled={isProcessing || isPlaying}
                            >
                                {isRecording ? '⏹️ Stop Recording' : '🎤 Start Recording'}
                            </button>
                        ) : (
                            <div className="results-section">
                                <h3>Conversation Complete</h3>
                                {score !== null && (
                                    <div className="score-display">
                                        <p>Depression Score: <strong>{score.toFixed(2)}</strong></p>
                                    </div>
                                )}
                                <button onClick={resetConversation} className="reset-button">
                                    🔄 Start New Conversation
                                </button>
                            </div>
                        )}
                    </div>
                </>
            )}
        </div>
    );
};

export default AudioChat;
