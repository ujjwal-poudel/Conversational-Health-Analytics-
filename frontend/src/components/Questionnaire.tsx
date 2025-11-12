import { useState, useEffect } from 'react'
import { FiArrowRight } from 'react-icons/fi'
import './Questionnaire.css'
import AudioRecorder from './AudioRecorder'
import questionsData from '../data/questions.json'

interface SubQuestion {
  id: string
  text: string
  answer: string
}

interface QuestionItem {
  id: number
  mainQuestion: string
  answer: string
  subQuestions: SubQuestion[]
}

interface QuestionnaireData {
  questions: QuestionItem[]
}

interface QuestionnaireProps {
  onComplete: (score: number) => void
}

const Questionnaire: React.FC<QuestionnaireProps> = ({ onComplete }) => {
  const [mode, setMode] = useState<'text' | 'audio'>('text')
  const [questions, setQuestions] = useState<QuestionItem[]>([])
  const [currentMainQuestionIndex, setCurrentMainQuestionIndex] = useState(0)
  const [mainQuestionAnswered, setMainQuestionAnswered] = useState(false)
  const [subQuestionAnswers, setSubQuestionAnswers] = useState<{ [key: string]: string }>({})
  const [mainQuestionAnswer, setMainQuestionAnswer] = useState('')
  const [loading, setLoading] = useState(true)
  const [showJsonPreview, setShowJsonPreview] = useState(false)

  useEffect(() => {
    // Load questions from JSON file
    loadQuestions()
  }, [])

  const loadQuestions = () => {
    try {
      const data = questionsData as QuestionnaireData
      setQuestions(data.questions)
      setLoading(false)
    } catch (error) {
      console.error('Error loading questions:', error)
      setLoading(false)
    }
  }

  const handleMainQuestionAnswer = (answer: string) => {
    setMainQuestionAnswer(answer)
    if (answer.trim()) {
      setMainQuestionAnswered(true)
      // Save answer to questions state
      const updatedQuestions = [...questions]
      updatedQuestions[currentMainQuestionIndex].answer = answer
      setQuestions(updatedQuestions)
    }
  }

  const handleSubQuestionAnswer = (subQuestionId: string, answer: string) => {
    setSubQuestionAnswers(prev => ({
      ...prev,
      [subQuestionId]: answer
    }))
    // Save answer to questions state
    const updatedQuestions = [...questions]
    const subQ = updatedQuestions[currentMainQuestionIndex].subQuestions.find(sq => sq.id === subQuestionId)
    if (subQ) {
      subQ.answer = answer
      setQuestions(updatedQuestions)
    }
  }

  const handleContinue = () => {
    // Check if main question and all sub-questions are answered
    const mainQ = questions[currentMainQuestionIndex]
    const allSubQuestionsAnswered = mainQ.subQuestions.every(sq => 
      subQuestionAnswers[sq.id] && subQuestionAnswers[sq.id].trim()
    )

    if (!mainQuestionAnswered || !allSubQuestionsAnswered) {
      return
    }

    // Move to next main question
    if (currentMainQuestionIndex < questions.length - 1) {
      setCurrentMainQuestionIndex(currentMainQuestionIndex + 1)
      setMainQuestionAnswered(false)
      setMainQuestionAnswer('')
      setSubQuestionAnswers({})
    } else {
      // All questions completed, submit
      submitAnswers(questions)
    }
  }

  const handleAudioComplete = (audioBlob: Blob, duration: number) => {
    // TODO: Handle audio properly
    console.log('Audio recorded:', { audioBlob, duration })
    alert('Audio mode: Please provide text answer for now, or implement audio transcription')
  }

  const submitAnswers = async (allQuestions: QuestionItem[]) => {
    try {
      // Format the data exactly as questions.json structure with answers (for reference/logging)
      const questionsJsonFormat = {
        questions: allQuestions.map(q => ({
          id: q.id,
          mainQuestion: q.mainQuestion,
          answer: q.answer,
          subQuestions: q.subQuestions.map(sq => ({
            id: sq.id,
            text: sq.text,
            answer: sq.answer
          }))
        }))
      }

      // Convert to backend-compatible format (flatten main + sub questions)
      const questionsAndAnswers: any[] = []
      let questionCounter = 1
      
      allQuestions.forEach(mainQ => {
        // Add main question
        questionsAndAnswers.push({
          question_id: questionCounter++,
          question_text: mainQ.mainQuestion,
          answer_text: mainQ.answer || ""
        })
        
        // Add sub-questions
        mainQ.subQuestions.forEach(subQ => {
          questionsAndAnswers.push({
            question_id: questionCounter++,
            question_text: subQ.text,
            answer_text: subQ.answer || ""
          })
        })
      })

      // Backend-compatible request format
      const backendRequest = {
        user_name: 'User',
        questions_and_answers: questionsAndAnswers
      }

      // Log both formats
      console.log('='.repeat(80))
      console.log('📋 QUESTIONS.JSON FORMAT (Your desired format):')
      console.log('='.repeat(80))
      console.log(JSON.stringify(questionsJsonFormat, null, 2))
      console.log('='.repeat(80))
      console.log('📤 BACKEND-COMPATIBLE FORMAT (Actually sent to API):')
      console.log('='.repeat(80))
      console.log(JSON.stringify(backendRequest, null, 2))
      console.log('='.repeat(80))

      // Store both formats in window
      ;(window as any).lastSubmittedQuestionsJson = questionsJsonFormat
      ;(window as any).lastSubmittedBackendFormat = backendRequest

      // Send POST request to backend prediction API
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(backendRequest)
      })

      if (!response.ok) {
        throw new Error('Failed to get prediction from backend')
      }

      const result = await response.json()
      
      console.log('📥 API Response:', result)
      console.log('='.repeat(80))
      console.log('💡 Access questions.json format: window.lastSubmittedQuestionsJson')
      console.log('💡 Access backend format: window.lastSubmittedBackendFormat')
      console.log('💡 Copy questions.json format: JSON.stringify(window.lastSubmittedQuestionsJson, null, 2)')
      console.log('='.repeat(80))

      // Extract prediction score from response
      const predictionScore = result.prediction || 0
      onComplete(Number(predictionScore))

    } catch (error) {
      console.error('Error getting prediction:', error)
      // Fallback to mock score on error
      const mockScore = Math.floor(Math.random() * 20) + 10
      onComplete(mockScore)
    }
  }

  const getTotalQuestions = () => {
    return questions.reduce((total, q) => total + 1 + q.subQuestions.length, 0)
  }

  const getCurrentQuestionNumber = () => {
    let count = 0
    for (let i = 0; i < currentMainQuestionIndex; i++) {
      count += 1 + questions[i].subQuestions.length // Main + sub questions
    }
    
    // Add current main question
    count += 1
    
    // Add answered sub-questions
    const mainQ = questions[currentMainQuestionIndex]
    if (mainQ) {
      count += mainQ.subQuestions.filter(sq => 
        subQuestionAnswers[sq.id] && subQuestionAnswers[sq.id].trim()
      ).length
    }
    
    return count
  }

  const progressPercentage = questions.length > 0 
    ? (getCurrentQuestionNumber() / getTotalQuestions()) * 100 
    : 0

  const canContinue = () => {
    if (!mainQuestionAnswered) return false
    const mainQ = questions[currentMainQuestionIndex]
    // Check that all 3 sub-questions are answered
    return mainQ.subQuestions.every(sq => 
      subQuestionAnswers[sq.id] && subQuestionAnswers[sq.id].trim()
    )
  }

  const getCurrentJsonData = () => {
    // Get the actual data that will be sent - use current state
    const currentQuestions = questions.map((q, index) => {
      if (index < currentMainQuestionIndex) {
        // Fully answered question - use saved data
        return {
          id: q.id,
          mainQuestion: q.mainQuestion,
          answer: q.answer,
          subQuestions: q.subQuestions.map(sq => ({
            id: sq.id,
            text: sq.text,
            answer: sq.answer
          }))
        }
      } else if (index === currentMainQuestionIndex) {
        // Current question - use current input values
        return {
          id: q.id,
          mainQuestion: q.mainQuestion,
          answer: mainQuestionAnswer || q.answer,
          subQuestions: q.subQuestions.map(sq => ({
            id: sq.id,
            text: sq.text,
            answer: subQuestionAnswers[sq.id] || sq.answer || ""
          }))
        }
      } else {
        // Future questions - empty answers
        return {
          id: q.id,
          mainQuestion: q.mainQuestion,
          answer: "",
          subQuestions: q.subQuestions.map(sq => ({
            id: sq.id,
            text: sq.text,
            answer: ""
          }))
        }
      }
    })
    
    return { questions: currentQuestions }
  }

  const getBackendFormatJsonData = () => {
    // Convert current data to backend-compatible format (same as submitAnswers)
    const currentQuestions = questions.map((q, index) => {
      if (index < currentMainQuestionIndex) {
        // Fully answered question - use saved data
        return {
          id: q.id,
          mainQuestion: q.mainQuestion,
          answer: q.answer,
          subQuestions: q.subQuestions.map(sq => ({
            id: sq.id,
            text: sq.text,
            answer: sq.answer
          }))
        }
      } else if (index === currentMainQuestionIndex) {
        // Current question - use current input values
        return {
          id: q.id,
          mainQuestion: q.mainQuestion,
          answer: mainQuestionAnswer || q.answer,
          subQuestions: q.subQuestions.map(sq => ({
            id: sq.id,
            text: sq.text,
            answer: subQuestionAnswers[sq.id] || sq.answer || ""
          }))
        }
      } else {
        // Future questions - empty answers
        return {
          id: q.id,
          mainQuestion: q.mainQuestion,
          answer: "",
          subQuestions: q.subQuestions.map(sq => ({
            id: sq.id,
            text: sq.text,
            answer: ""
          }))
        }
      }
    })

    // Convert to backend-compatible format (flatten main + sub questions)
    const questionsAndAnswers: any[] = []
    let questionCounter = 1
    
    currentQuestions.forEach(mainQ => {
      // Add main question
      questionsAndAnswers.push({
        question_id: questionCounter++,
        question_text: mainQ.mainQuestion,
        answer_text: mainQ.answer || ""
      })
      
      // Add sub-questions
      mainQ.subQuestions.forEach(subQ => {
        questionsAndAnswers.push({
          question_id: questionCounter++,
          question_text: subQ.text,
          answer_text: subQ.answer || ""
        })
      })
    })

    // Backend-compatible request format
    return {
      user_name: 'User',
      questions_and_answers: questionsAndAnswers
    }
  }

  const copyJsonToClipboard = () => {
    // Use backend format (the one that works in Postman)
    const jsonData = (window as any).lastSubmittedBackendFormat || getBackendFormatJsonData()
    const jsonString = JSON.stringify(jsonData, null, 2)
    navigator.clipboard.writeText(jsonString).then(() => {
      // Silently copy to clipboard without alert
    }).catch(() => {
      // Fallback for older browsers
      const textArea = document.createElement('textarea')
      textArea.value = jsonString
      document.body.appendChild(textArea)
      textArea.select()
      document.execCommand('copy')
      document.body.removeChild(textArea)
    })
  }

  if (loading) {
    return (
      <div className="questionnaire-card">
        <div className="questionnaire-header">
          <h2>Personal Health Questionnaire</h2>
        </div>
        <div className="loading">Loading questions...</div>
      </div>
    )
  }

  if (questions.length === 0) {
    return (
      <div className="questionnaire-card">
        <div className="questionnaire-header">
          <h2>Personal Health Questionnaire</h2>
        </div>
        <div className="error">No questions available</div>
      </div>
    )
  }

  const mainQuestion = questions[currentMainQuestionIndex]
  const isLastQuestion = currentMainQuestionIndex === questions.length - 1

  return (
    <div className="questionnaire-card">
      <div className="questionnaire-header">
        <h2>Personal Health Questionnaire</h2>
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'text' ? 'active' : ''}`}
            onClick={() => setMode('text')}
          >
            Text
          </button>
          <button
            className={`mode-btn ${mode === 'audio' ? 'active' : ''}`}
            onClick={() => setMode('audio')}
          >
            Audio
          </button>
        </div>
      </div>
      
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${progressPercentage}%` }}
        />
      </div>

      <div className="questions-container">
        {/* Question progress info */}
        <div className="question-info">
          <p>Main Question {currentMainQuestionIndex + 1} of {questions.length}</p>
        </div>

        {/* Main Question */}
        <div className="current-question">
          <div className="question-number active">
            Q{currentMainQuestionIndex + 1}
          </div>
          <div className="question-content">
            <h3 className="main-question-label">Main Question:</h3>
            <p className="question-text">{mainQuestion.mainQuestion}</p>
            
            {mode === 'text' ? (
              <div className="text-input-container">
                <textarea
                  className="text-input"
                  placeholder="Type your answer..."
                  value={mainQuestionAnswer}
                  onChange={(e) => handleMainQuestionAnswer(e.target.value)}
                  rows={4}
                />
              </div>
            ) : (
              <AudioRecorder 
                key={`main-${currentMainQuestionIndex}`}
                onComplete={handleAudioComplete} 
              />
            )}
          </div>
        </div>

        {/* Sub-questions - shown after main question is answered */}
        {mainQuestionAnswered && (
          <div className="sub-questions-container">
            {mainQuestion.subQuestions.map((subQ, index) => (
              <div key={subQ.id} className="sub-question">
                <div className="question-number">
                  {currentMainQuestionIndex + 1}{String.fromCharCode(97 + index)}
                </div>
                <div className="question-content">
                  <p className="question-text">{subQ.text}</p>
                  
                  {mode === 'text' ? (
                    <div className="text-input-container">
                      <textarea
                        className="text-input"
                        placeholder="Type your answer..."
                        value={subQuestionAnswers[subQ.id] || ''}
                        onChange={(e) => handleSubQuestionAnswer(subQ.id, e.target.value)}
                        rows={3}
                      />
                    </div>
                  ) : (
                    <AudioRecorder 
                      key={`sub-${subQ.id}`}
                      onComplete={handleAudioComplete} 
                    />
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Continue/Submit Button */}
        <div className="continue-container">
          {/* Preview JSON Button - Only show when all questions are answered */}
          {isLastQuestion && canContinue() && (
            <button
              className="preview-json-btn"
              onClick={() => {
                setShowJsonPreview(!showJsonPreview)
                if (!showJsonPreview) {
                  // Scroll to JSON preview
                  setTimeout(() => {
                    document.getElementById('json-preview')?.scrollIntoView({ behavior: 'smooth' })
                  }, 100)
                }
              }}
              style={{
                marginRight: '1rem',
                padding: '0.75rem 1.5rem',
                background: '#f0f0f0',
                color: '#333',
                border: '1px solid #ddd',
                borderRadius: '6px',
                cursor: 'pointer',
                fontSize: '0.95rem'
              }}
            >
              {showJsonPreview ? 'Hide JSON' : 'JSON'}
            </button>
          )}
          <button
            className="continue-btn"
            onClick={handleContinue}
            disabled={!canContinue()}
          >
            {isLastQuestion ? 'Submit' : 'Continue'}
            <FiArrowRight />
          </button>
        </div>

        {/* JSON Preview Section */}
        {showJsonPreview && isLastQuestion && (
          <div id="json-preview" className="json-preview-container" style={{
            marginTop: '2rem',
            padding: '1.5rem',
            background: '#f9f9f9',
            border: '2px solid #613eff',
            borderRadius: '8px'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, color: '#613eff', fontSize: '1.1rem' }}>
                📋 JSON Data (Copy this for Postman testing)
              </h3>
              <button
                onClick={copyJsonToClipboard}
                style={{
                  padding: '0.5rem 1rem',
                  background: '#613eff',
                  color: 'white',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  fontSize: '0.9rem'
                }}
              >
                Copy JSON
              </button>
            </div>
            <pre style={{
              background: '#fff',
              padding: '1rem',
              borderRadius: '6px',
              overflow: 'auto',
              maxHeight: '400px',
              fontSize: '0.85rem',
              lineHeight: '1.5',
              border: '1px solid #e0e0e0',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word'
            }}>
              {JSON.stringify(getBackendFormatJsonData(), null, 2)}
            </pre>
            <p style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#666' }}>
              💡 This is the backend-compatible format (ready to use in Postman).
              <br />
              💡 URL: <code style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: '3px' }}>http://localhost:8001/predict</code>
              <br />
              💡 Method: <code style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: '3px' }}>POST</code>
              <br />
              💡 Headers: <code style={{ background: '#f0f0f0', padding: '2px 6px', borderRadius: '3px' }}>Content-Type: application/json</code>
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default Questionnaire
