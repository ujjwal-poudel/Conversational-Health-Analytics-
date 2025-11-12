import { useState } from 'react'
import Layout from './components/Layout'
import Questionnaire from './components/Questionnaire'
import Results from './components/Results'
import './App.css'

function App() {
  const [showResults, setShowResults] = useState(false)
  const [depressionScore, setDepressionScore] = useState(0)

  const handleQuestionnaireComplete = (score: number) => {
    setDepressionScore(score)
    setShowResults(true)
  }

  return (
    <Layout>
      {showResults ? (
        <Results score={depressionScore} />
      ) : (
        <Questionnaire onComplete={handleQuestionnaireComplete} />
      )}
    </Layout>
  )
}

export default App

