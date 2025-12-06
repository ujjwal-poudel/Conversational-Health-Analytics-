import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import GradientOrb from '../components/GradientOrb';
import './ModeSelect.css';

type Mode = 'text' | 'audio';

const ModeSelect: React.FC = () => {
    const navigate = useNavigate();
    const [selectedMode, setSelectedMode] = useState<Mode>('text');

    const handleContinue = () => {
        if (selectedMode === 'text') {
            navigate('/chat');
        } else {
            navigate('/audio');
        }
    };

    return (
        <div className="mode-select">
            <div className="mode-select__header">
                <button className="mode-select__back-orb" onClick={() => navigate(-1)} aria-label="Go back" title="Go back">
                    <GradientOrb size="small" />
                </button>
            </div>

            <div className="mode-select__content">
                <h1 className="mode-select__title">
                    <span className="mode-select__greeting">Hey there!</span>
                    {' '}Choose a communication mode for assessment?
                </h1>

                <div className="mode-select__toggle">
                    <button
                        className={`mode-select__option ${selectedMode === 'text' ? 'mode-select__option--active' : ''}`}
                        onClick={() => setSelectedMode('text')}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                        </svg>
                        Text
                    </button>
                    <button
                        className={`mode-select__option ${selectedMode === 'audio' ? 'mode-select__option--active' : ''}`}
                        onClick={() => setSelectedMode('audio')}
                    >
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                            <line x1="12" y1="19" x2="12" y2="23" />
                            <line x1="8" y1="23" x2="16" y2="23" />
                        </svg>
                        Audio
                    </button>
                </div>

                <button className="mode-select__continue" onClick={handleContinue}>
                    Continue
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                </button>
            </div>
        </div>
    );
};

export default ModeSelect;
