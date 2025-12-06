import React from 'react';
import './GradientOrb.css';

interface GradientOrbProps {
    size?: 'small' | 'large';
    className?: string;
}

const GradientOrb: React.FC<GradientOrbProps> = ({ size = 'large', className = '' }) => {
    return (
        <div className={`gradient-orb gradient-orb--${size} ${className}`}>
            <div className="gradient-orb__inner" />
            <div className="gradient-orb__glow" />
        </div>
    );
};

export default GradientOrb;
