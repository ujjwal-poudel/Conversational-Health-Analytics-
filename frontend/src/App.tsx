import { useState } from 'react'
import Layout from './components/Layout'
import Questionnaire from './components/Questionnaire'
import Chatbot from './components/Chatbot'
import Results from './components/Results'
import './App.css'

function App() {
  const [view, setView] = useState<'questionnaire' | 'chatbot'>('chatbot')
  const [showResults, setShowResults] = useState(false)
  const [depressionScore, setDepressionScore] = useState(0)

  const handleQuestionnaireComplete = (score: number) => {
    setDepressionScore(score)
    setShowResults(true)
  }

  return (
    <Layout>
      <div style={{ marginBottom: '20px', display: 'flex', justifyContent: 'center', gap: '10px' }}>
        <button
          onClick={() => setView('chatbot')}
          style={{
            padding: '8px 16px',
            background: view === 'chatbot' ? '#764ba2' : 'rgba(255,255,255,0.1)',
            border: 'none',
            borderRadius: '20px',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          AI Chatbot
        </button>
        <button
          onClick={() => setView('questionnaire')}
          style={{
            padding: '8px 16px',
            background: view === 'questionnaire' ? '#764ba2' : 'rgba(255,255,255,0.1)',
            border: 'none',
            borderRadius: '20px',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          Questionnaire
        </button>
      </div>

      {view === 'chatbot' ? (
        <Chatbot />
      ) : (
        showResults ? (
          <Results score={depressionScore} />
        ) : (
          <Questionnaire onComplete={handleQuestionnaireComplete} />
        )
      )}
    </Layout>
  )
}

export default App

