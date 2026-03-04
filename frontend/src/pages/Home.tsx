import React from 'react';
import { useNavigate } from 'react-router-dom';
import GradientOrb from '../components/GradientOrb';
import './Home.css';

const Home: React.FC = () => {
    const navigate = useNavigate();

    const handleGetStarted = () => {
        navigate('/mode');
    };

    const handleAbout = () => {
        navigate('/about');
    };

    return (
        <div className="home">
            <div className="home__content">
                <div className="home__text">
                    <h1 className="home__title">Conversational Health Analytics</h1>
                    <p className="home__tagline">
                        Mental health assessment,<br />
                        at your convenience!
                    </p>
                    <div className="home__actions">
                        <button
                            className="home__cta"
                            onClick={handleGetStarted}
                        >
                            Get started!
                            <svg
                                width="20"
                                height="20"
                                viewBox="0 0 24 24"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                            >
                                <path d="M5 12h14M12 5l7 7-7 7" />
                            </svg>
                        </button>
                    </div>
                </div>

                <div className="home__orb">
                    <GradientOrb size="large" />
                </div>
            </div>

            <span
                className="home__about-link"
                onClick={handleAbout}
            >
                Learn More
            </span>
        </div>
    );
};

export default Home;
