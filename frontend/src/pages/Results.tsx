import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import GradientOrb from '../components/GradientOrb';
import './Results.css';

const MAX_SCORE = 24;
const THRESHOLD = 10;
const ANIMATION_DURATION = 2000; // 2 seconds for full animation

const Results: React.FC = () => {
    const navigate = useNavigate();
    const location = useLocation();

    // Get score from navigation state
    const finalScore = location.state?.score ?? null;

    const [displayScore, setDisplayScore] = useState(0);
    const [isAnimating, setIsAnimating] = useState(true);
    const [showMessage, setShowMessage] = useState(false);
    const [showHomeButton, setShowHomeButton] = useState(false);

    // Animate the score
    useEffect(() => {
        if (finalScore === null) {
            // Still loading
            return;
        }

        const targetScore = Math.round(finalScore);
        const startTime = Date.now();
        const duration = ANIMATION_DURATION;

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Easing function for smooth animation
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const currentScore = Math.round(easeOutQuart * targetScore);

            setDisplayScore(currentScore);

            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                setDisplayScore(targetScore);
                setIsAnimating(false);
                // Show message after score animation completes
                setTimeout(() => setShowMessage(true), 300);
                // Show home button after message
                setTimeout(() => setShowHomeButton(true), 800);
            }
        };

        // Small delay before starting animation
        setTimeout(() => {
            requestAnimationFrame(animate);
        }, 500);
    }, [finalScore]);

    // Calculate circle progress (0 to 1)
    const progress = finalScore !== null ? displayScore / MAX_SCORE : 0;

    // Circle dimensions
    const size = 220;
    const strokeWidth = 12;
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDashoffset = circumference * (1 - progress);

    // Get color based on score
    const getColor = (score: number): string => {
        if (score < THRESHOLD) {
            return '#22c55e'; // Green
        } else {
            // Red with increasing intensity
            const intensity = Math.min((score - THRESHOLD) / (MAX_SCORE - THRESHOLD), 1);
            const red = 220;
            const green = Math.round(50 * (1 - intensity));
            const blue = Math.round(50 * (1 - intensity));
            return `rgb(${red}, ${green}, ${blue})`;
        }
    };

    const strokeColor = getColor(displayScore);

    // Get message based on score
    const getMessage = (): { title: string; subtitle: string } => {
        if (finalScore === null || isAnimating) {
            return { title: '', subtitle: '' };
        }

        if (finalScore < THRESHOLD) {
            return {
                title: "Great news!",
                subtitle: "You do not show significant signs of depression. Keep taking care of your mental health!"
            };
        } else {
            return {
                title: "We recommend seeking support",
                subtitle: "You have a high chance of mental depression, please visit a mental health professional for consultation."
            };
        }
    };

    const message = getMessage();

    return (
        <div className="results">
            <div className="results__orb">
                <GradientOrb size="small" />
            </div>

            <div className="results__content">
                <h1 className="results__title">Your risk score is here!</h1>

                <div className="results__circle-container">
                    <svg
                        className="results__circle"
                        width={size}
                        height={size}
                        viewBox={`0 0 ${size} ${size}`}
                    >
                        {/* Background circle */}
                        <circle
                            cx={size / 2}
                            cy={size / 2}
                            r={radius}
                            fill="none"
                            stroke="#1e293b"
                            strokeWidth={strokeWidth}
                        />
                        {/* Progress circle */}
                        <circle
                            cx={size / 2}
                            cy={size / 2}
                            r={radius}
                            fill="none"
                            stroke={strokeColor}
                            strokeWidth={strokeWidth}
                            strokeLinecap="round"
                            strokeDasharray={circumference}
                            strokeDashoffset={strokeDashoffset}
                            transform={`rotate(-90 ${size / 2} ${size / 2})`}
                            style={{ transition: 'stroke 0.3s ease' }}
                        />
                    </svg>

                    <div className="results__score" style={{ color: strokeColor }}>
                        {finalScore === null ? (
                            <span className="results__assessing">Assessing...</span>
                        ) : (
                            displayScore
                        )}
                    </div>
                </div>

                <div className={`results__message ${showMessage ? 'visible' : ''}`}>
                    <p className="results__message-text">
                        {message.subtitle}
                    </p>
                </div>

                <button
                    className={`results__home-button ${showHomeButton ? 'visible' : ''}`}
                    onClick={() => navigate('/')}
                >
                    Back to Home
                </button>
            </div>
        </div>
    );
};

export default Results;
