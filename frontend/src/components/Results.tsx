import React from 'react'
import './Results.css'

interface ResultsProps {
  score: number
}

const Results: React.FC<ResultsProps> = ({ score }) => {
  // Calculate the gauge angle (0-180 degrees)
  // Score range: 0-27 (standard PHQ-9 range)
  // But for visual purposes, we'll map it to a percentage
  const maxScore = 27
  const percentage = Math.min((score / maxScore) * 100, 100)
  const rotation = (percentage / 100) * 180 - 90 // -90 to 90 degrees

  // Determine severity level
  const getSeverityLevel = () => {
    if (score >= 20) return { level: 'Severe', color: '#d32f2f' }
    if (score >= 15) return { level: 'Moderately Severe', color: '#f57c00' }
    if (score >= 10) return { level: 'Moderate', color: '#ffa726' }
    if (score >= 5) return { level: 'Mild', color: '#ffb74d' }
    return { level: 'Minimal', color: '#66bb6a' }
  }

  const { level, color } = getSeverityLevel()

  return (
    <div className="results-card">
      <div className="results-header">
        <h2>Personal Health Questionnaire</h2>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: '100%' }} />
        </div>
      </div>

      <div className="results-content">
        <p className="results-intro">Based on your answers, here are the results...</p>

        <div className="gauge-container">
          <svg className="gauge-svg" viewBox="0 0 200 120" width="300" height="180">
            {/* Background arc (grey) */}
            <path
              d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none"
              stroke="#e0e0e0"
              strokeWidth="20"
              strokeLinecap="round"
            />
            
            {/* Colored arc (red for depression) */}
            <path
              d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none"
              stroke={color}
              strokeWidth="20"
              strokeLinecap="round"
              strokeDasharray={`${percentage * 2.51} 251`}
              style={{ transition: 'stroke-dasharray 1s ease-out' }}
            />
            
            {/* Center text */}
            <text
              x="100"
              y="90"
              textAnchor="middle"
              fontSize="48"
              fontWeight="600"
              fill="#333"
            >
              {score.toFixed(2)}
            </text>
          </svg>
          
          <p className="gauge-label">Depression severity index</p>
        </div>

        <div className="results-message">
          <p>
            Our model suggests that you are <strong style={{ color }}>{level.toLowerCase()}</strong>!
            {score >= 10 && (
              <> We suggest you to <strong>visit a doctor</strong> to consult your condition!</>
            )}
          </p>
        </div>

        <div className="severity-info">
          <h3>Understanding Your Score</h3>
          <ul>
            <li><span className="severity-range">0-4:</span> Minimal depression</li>
            <li><span className="severity-range">5-9:</span> Mild depression</li>
            <li><span className="severity-range">10-14:</span> Moderate depression</li>
            <li><span className="severity-range">15-19:</span> Moderately severe depression</li>
            <li><span className="severity-range">20-27:</span> Severe depression</li>
          </ul>
          <p className="disclaimer">
            <strong>Note:</strong> This is a screening tool and not a diagnostic instrument. 
            Please consult with a healthcare professional for proper diagnosis and treatment.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Results

