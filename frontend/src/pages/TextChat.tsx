import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { getActiveApiUrl } from '../config/api';
import GradientOrb from '../components/GradientOrb';
import './TextChat.css';

interface Message {
    sender: 'bot' | 'user';
    text: string;
}

interface ChatResponse {
    response: string | string[];
    is_finished: boolean;
    depression_score: number | null;
    session_id: string | null;
}

const TextChat: React.FC = () => {
    const navigate = useNavigate();
    const API_BASE_URL = `${getActiveApiUrl()}/api/v1/chat`;
    const [messages, setMessages] = useState<Message[]>([]);
    const [inputText, setInputText] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isTyping, setIsTyping] = useState(false);
    const [isFinished, setIsFinished] = useState(false);
    const [_depressionScore, setDepressionScore] = useState<number | null>(null);


    const messagesEndRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const typingIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const hasStartedRef = useRef(false);
    const sessionIdRef = useRef<string | null>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        if (!isTyping && !isLoading && !isFinished) {
            inputRef.current?.focus();
        }
    }, [isTyping, isLoading, isFinished]);

    // Start chat on mount - prevent double calls
    useEffect(() => {
        if (hasStartedRef.current) return;
        hasStartedRef.current = true;
        startChat();

        return () => {
            // Cleanup on unmount
            if (sessionIdRef.current) {
                navigator.sendBeacon(
                    `${API_BASE_URL}/cleanup`,
                    JSON.stringify({ session_id: sessionIdRef.current })
                );
            }
        };
    }, []);

    // Handle browser back button
    useEffect(() => {
        const handlePopState = () => {
            // Cleanup chat when user presses browser back
            if (sessionIdRef.current) {
                navigator.sendBeacon(
                    `${API_BASE_URL}/cleanup`,
                    JSON.stringify({ session_id: sessionIdRef.current })
                );
            }
        };

        window.addEventListener('popstate', handlePopState);
        return () => window.removeEventListener('popstate', handlePopState);
    }, []);

    // Handle back button click
    const handleBack = async () => {
        try {
            await axios.post(`${API_BASE_URL}/cleanup`, { session_id: sessionIdRef.current });
        } catch (e) {
            console.error('Cleanup error:', e);
        }
        navigate(-1);
    };

    const typeMessage = (text: string, callback?: () => void) => {
        setIsTyping(true);
        let index = 0;
        let currentText = '';

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
                    if (newMessages[lastIndex]?.sender === 'bot') {
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
        }, 15);
    };

    const typeMultiPartMessage = (parts: string[], callback?: () => void) => {
        let currentPartIndex = 0;

        const typeNextPart = () => {
            if (currentPartIndex < parts.length) {
                typeMessage(parts[currentPartIndex], () => {
                    currentPartIndex++;
                    if (currentPartIndex < parts.length) {
                        setTimeout(typeNextPart, 400);
                    } else {
                        if (callback) callback();
                    }
                });
            }
        };

        typeNextPart();
    };

    const startChat = async () => {
        setIsLoading(true);
        try {
            const res = await axios.post<ChatResponse>(`${API_BASE_URL}/start`);
            // Store server-generated session ID
            if (res.data.session_id) {
                sessionIdRef.current = res.data.session_id;
            }
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
    };

    const sendMessage = async () => {
        if (!inputText.trim() || isTyping || isLoading) return;

        const userMsg = inputText;
        setMessages(prev => [...prev, { sender: 'user', text: userMsg }]);
        setInputText('');
        setIsLoading(true);

        try {
            const res = await axios.post<ChatResponse>(`${API_BASE_URL}/message`, {
                message: userMsg,
                session_id: sessionIdRef.current,
            });

            const responseParts = Array.isArray(res.data.response)
                ? res.data.response
                : [res.data.response];

            typeMultiPartMessage(responseParts, () => {
                if (res.data.is_finished) {
                    setIsFinished(true);
                    setDepressionScore(res.data.depression_score);


                    // Wait 3 seconds for user to read final message, then navigate to results
                    setTimeout(() => {
                        navigate('/results', { state: { score: res.data.depression_score } });
                    }, 3000);
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

    // Unused function - commented out
    // const getResultContent = () => {
    //     if (depressionScore === null) return null;

    //     const isDepressed = depressionScore >= 10;
    //     const mainText = isDepressed
    //         ? "Our AI model suggests you may have signs of depression."
    //         : "Our AI model suggests you don't show significant signs of depression.";

    //     let semanticText = "";
    //     if (consistencyStatus === 'conflict_potential_false_negative') {
    //         semanticText = "However, our semantic analysis detected patterns similar to those who experience depression.";
    //     } else if (consistencyStatus === 'conflict_potential_false_positive') {
    //         semanticText = "Our semantic analysis suggests your conversation style aligns with healthy patterns.";
    //     } else if (consistencyStatus === 'strong_agreement') {
    //         semanticText = "Our semantic analysis confirms this finding.";
    //     }

    //     return (
    //         <div className="result__content">
    //             <p className="result__main-text">{mainText}</p>
    //             {semanticText && <p className="result__semantic-text">{semanticText}</p>}
    //         </div>
    //     );
    // };

    return (
        <div className="text-chat">
            <div className="text-chat__header">
                <button className="text-chat__back-orb" onClick={handleBack} aria-label="Go back" title="Go back">
                    <GradientOrb size="small" />
                </button>
            </div>

            <div className="text-chat__messages">
                {messages.map((msg, index) => (
                    <div
                        key={index}
                        className={`message message--${msg.sender}`}
                    >
                        {msg.text}
                    </div>
                ))}
                {isLoading && !isTyping && (
                    <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            <div className="text-chat__input-area">
                <input
                    ref={inputRef}
                    type="text"
                    className="text-chat__input"
                    placeholder="Ask anything...."
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    onKeyDown={handleKeyPress}
                    disabled={isLoading || isFinished || isTyping}
                />
                <button
                    className="text-chat__send"
                    onClick={sendMessage}
                    disabled={!inputText.trim() || isLoading || isFinished || isTyping}
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M12 19V5M5 12l7-7 7 7" />
                    </svg>
                </button>
            </div>
        </div>
    );
};

export default TextChat;
