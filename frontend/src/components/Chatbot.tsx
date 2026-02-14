import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { FaPaperPlane, FaRobot, FaUser } from 'react-icons/fa';
import { getActiveApiUrl } from '../config/api';
import './Chatbot.css';

interface Message {
    sender: 'bot' | 'user';
    text: string;
}

interface ChatResponse {
    response: string | string[];  // Support both single string and array
    is_finished: boolean;
    depression_score: number | null;
    semantic_risk_label: string | null;
    consistency_status: string | null;
}

const Chatbot = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isStarted, setIsStarted] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [depressionScore, setDepressionScore] = useState<number | null>(null);
    const [consistencyStatus, setConsistencyStatus] = useState<string | null>(null);
    const [isTyping, setIsTyping] = useState(false);
    const API_BASE_URL = `${getActiveApiUrl()}/api/v1/chat`;

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Auto-focus input when typing completes
    useEffect(() => {
        if (!isTyping && !isLoading && !isFinished && isStarted) {
            inputRef.current?.focus();
        }
    }, [isTyping, isLoading, isFinished, isStarted]);

    // Cleanup on page refresh/close if conversation is active but not finished
    useEffect(() => {
        const handleBeforeUnload = () => {
            if (isStarted && !isFinished) {
                // sendBeacon is more reliable than fetch during page unload
                navigator.sendBeacon(`${API_BASE}/api/v1/chat/cleanup`);
            }
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        return () => window.removeEventListener('beforeunload', handleBeforeUnload);
    }, [isStarted, isFinished]);

    // Typing animation function
    const typeMessage = (text: string, callback?: () => void) => {
        setIsTyping(true);
        let index = 0;
        let currentText = '';

        // Add empty message that will be filled
        setMessages(prev => [...prev, { sender: 'bot', text: '' }]);

        if (typingIntervalRef.current) {
            clearInterval(typingIntervalRef.current);
        }

        typingIntervalRef.current = setInterval(() => {
            if (index < text.length) {
                currentText += text[index];
                setMessages(prev => {
                    const newMessages = [...prev];
                    const lastIndex = newMessages.length - 1;
                    if (newMessages[lastIndex] && newMessages[lastIndex].sender === 'bot') {
                        newMessages[lastIndex] = { ...newMessages[lastIndex], text: currentText };
                    }
                    return newMessages;
                });
                index++;
            } else {
                if (typingIntervalRef.current) {
                    clearInterval(typingIntervalRef.current);
                    typingIntervalRef.current = null;
                }
                setIsTyping(false);
                if (callback) callback();
            }
        }, 20); // 20ms per character for smooth typing
    };

    // Type multiple message parts sequentially with delays
    const typeMultiPartMessage = (parts: string[], callback?: () => void) => {
        let currentPartIndex = 0;

        const typeNextPart = () => {
            if (currentPartIndex < parts.length) {
                typeMessage(parts[currentPartIndex], () => {
                    currentPartIndex++;
                    if (currentPartIndex < parts.length) {
                        // Add delay between parts (600ms for natural pause)
                        setTimeout(typeNextPart, 600);
                    } else {
                        // All parts typed, call final callback
                        if (callback) callback();
                    }
                });
            }
        };

        typeNextPart();
    };

    const startChat = async () => {
        setIsLoading(true);
        // 1. Show Intro Message with typing effect
        const introMsg = "Hello I am diagnoser, I can use sophisticated ai to detect if you have any sign of depression, feel free to open up and answer as much as you can to get precise results.";
        setIsStarted(true);
        setIsFinished(false);
        setDepressionScore(null);
        setConsistencyStatus(null);

        typeMessage(introMsg, async () => {
            try {
                // 2. Fetch First Question from Backend
                const res = await axios.post<ChatResponse>(`${API_BASE_URL}/start`);

                // 3. Append Backend Question with typing animation
                const responseParts = Array.isArray(res.data.response)
                    ? res.data.response
                    : [res.data.response];
                typeMultiPartMessage(responseParts);
            } catch (error) {
                console.error("Error starting chat:", error);
                typeMessage("Sorry, I couldn't start the conversation. Please try again.");
            } finally {
                setIsLoading(false);
            }
        });
    };

    const sendMessage = async () => {
        if (!inputText.trim() || isTyping) return;

        const userMsg = inputText;
        setMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
        setInputText('');
        setIsLoading(true);

        try {
            const res = await axios.post<ChatResponse>(`${API_BASE_URL}/message`, {
                message: userMsg
            });

            // Handle both single string and array responses
            const responseParts = Array.isArray(res.data.response)
                ? res.data.response
                : [res.data.response];

            // Use typing animation for bot response with multi-part support
            typeMultiPartMessage(responseParts, () => {
                if (res.data.is_finished) {
                    setIsFinished(true);
                    setDepressionScore(res.data.depression_score);
                    setConsistencyStatus(res.data.consistency_status);
                }
            });
        } catch (error) {
            console.error("Error sending message:", error);
            typeMessage("Sorry, something went wrong. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    // Helper to generate the detailed result message
    const getResultMessage = () => {
        if (depressionScore === null) return null;

        const isDepressed = depressionScore >= 10;
        const scoreText = isDepressed
            ? "Our Artificial Intelligence Model predicts that you may have signs of depression."
            : "Our Artificial Intelligence Model predicts that you do not have any signs of depression.";

        let semanticText = "";

        if (consistencyStatus === 'conflict_potential_false_negative') {
            semanticText = "However, our semantic analysis detected that your conversation style is similar to the style of people who were categorized as depressed. This suggests a potential risk despite the low score.";
        } else if (consistencyStatus === 'conflict_potential_false_positive') {
            semanticText = "However, our semantic analysis suggests your conversation style is more similar to healthy individuals. This might indicate a false positive, but we recommend consulting a professional to be sure.";
        } else if (consistencyStatus === 'strong_agreement') {
            semanticText = isDepressed
                ? "Our semantic analysis also confirms this finding, showing a high similarity to patterns associated with depression."
                : "Our semantic analysis also confirms this, showing your conversation style aligns with healthy patterns.";
        }

        return (
            <div style={{ textAlign: 'left', marginTop: '20px' }}>
                <p style={{ marginBottom: '10px', fontSize: '1.1rem' }}>{scoreText}</p>
                {semanticText && (
                    <div style={{
                        padding: '15px',
                        background: 'rgba(255, 255, 255, 0.05)',
                        borderRadius: '10px',
                        borderLeft: '4px solid #764ba2',
                        marginTop: '15px'
                    }}>
                        <p style={{ margin: 0, fontSize: '0.95rem', color: '#ddd' }}>
                            <strong>Semantic Analysis:</strong> {semanticText}
                        </p>
                    </div>
                )}
            </div>
        );
    };

    if (!isStarted) {
        return (
            <div className="chatbot-container">
                <div className="start-screen">
                    <FaRobot size={60} color="#764ba2" style={{ marginBottom: '20px' }} />
                    <h2>Virtual Health Assistant</h2>
                    <p style={{ color: '#aaa', maxWidth: '400px', margin: '10px 0 30px' }}>
                        I'm here to have a friendly chat with you. Your responses are confidential and will help us understand your well-being.
                    </p>
                    <button className="start-button" onClick={startChat} disabled={isLoading}>
                        {isLoading ? 'Initializing...' : 'Start Conversation'}
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="chatbot-container">
            <div className="chatbot-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <FaRobot size={24} color="#764ba2" />
                    <h2>Health Assistant</h2>
                </div>
                {isFinished && <span style={{ color: '#4caf50', fontSize: '0.9rem' }}>Completed</span>}
            </div>

            <div className="messages-area">
                {messages.map((msg, index) => (
                    <div key={index} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start', alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%' }}>
                        {msg.sender === 'bot' && (
                            <div style={{
                                background: '#2d2d2d',
                                borderRadius: '50%',
                                padding: '8px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                minWidth: '32px',
                                height: '32px'
                            }}>
                                <FaRobot size={16} color="#8ab4f8" />
                            </div>
                        )}
                        <div className={`message ${msg.sender}`}>
                            {msg.text}
                        </div>
                        {msg.sender === 'user' && (
                            <div style={{
                                background: '#8ab4f8',
                                borderRadius: '50%',
                                padding: '8px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                minWidth: '32px',
                                height: '32px'
                            }}>
                                <FaUser size={16} color="#1e1e1e" />
                            </div>
                        )}
                    </div>
                ))}
                {isLoading && (
                    <div className="typing-indicator">
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="input-area">
                <input
                    ref={inputRef}
                    type="text"
                    className="chat-input"
                    placeholder="Type your answer..."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyPress}
                    disabled={isLoading || isFinished || isTyping}
                    autoFocus
                />
                <button
                    className="send-button"
                    onClick={sendMessage}
                    disabled={!inputText.trim() || isLoading || isFinished || isTyping}
                >
                    <FaPaperPlane size={18} />
                </button>
            </div>

            {isFinished && depressionScore !== null && (
                <div className="result-overlay">
                    <div className="result-card" style={{ maxWidth: '600px', width: '90%' }}>
                        <h3>Assessment Complete</h3>

                        <div className="score-display">
                            {depressionScore.toFixed(1)}
                        </div>
                        <p style={{ color: '#aaa', fontSize: '0.9rem', marginBottom: '20px' }}>
                            Depression Severity Score (PHQ-8)<br />
                            <span style={{ fontSize: '0.8rem', opacity: 0.8 }}>
                                Range: 0-24 | Threshold: 10
                            </span>
                        </p>

                        {getResultMessage()}

                        <button className="start-button" onClick={startChat} style={{ marginTop: '30px' }}>
                            Start New Session
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Chatbot;
