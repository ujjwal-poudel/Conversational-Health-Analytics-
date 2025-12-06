import React from 'react';
import { useNavigate } from 'react-router-dom';
import GradientOrb from '../components/GradientOrb';
import './Home.css';

const Home: React.FC = () => {
    const navigate = useNavigate();

    const handleGetStarted = () => {
        navigate('/mode');
    };

    return (
        <div className="home">
            <div className="home__content">
                <div className="home__text">
                    <h1 className="home__title">Dspresso.</h1>
                    <p className="home__tagline">
                        Mental health assessment,<br />
                        at your convenience!
                    </p>
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

                <div className="home__orb">
                    <GradientOrb size="large" />
                </div>
            </div>
        </div>
    );
};

export default Home;
