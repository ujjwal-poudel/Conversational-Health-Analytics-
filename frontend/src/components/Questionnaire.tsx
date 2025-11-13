import { useState, useEffect } from 'react'
import { FiArrowRight } from 'react-icons/fi'
import './Questionnaire.css'
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
  const [questions, setQuestions] = useState<QuestionItem[]>([])
  const [currentMainQuestionIndex, setCurrentMainQuestionIndex] = useState(0)
  const [mainQuestionAnswered, setMainQuestionAnswered] = useState(false)
  const [subQuestionAnswers, setSubQuestionAnswers] = useState<Record<string, string>>({})
  const [mainQuestionAnswer, setMainQuestionAnswer] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Load questions from JSON file - NO API CALL
    loadQuestions()
  }, [])

  const loadQuestions = () => {
    try {
      const data = questionsData as QuestionnaireData
      // Initialize all answers as empty strings
      const initializedQuestions = data.questions.map(q => ({
        ...q,
        answer: '',
        subQuestions: q.subQuestions.map(sq => ({ ...sq, answer: '' }))
      }))
      setQuestions(initializedQuestions)
      setLoading(false)
    } catch (error) {
      console.error('Error loading questions:', error)
      setLoading(false)
    }
  }

  const handleMainQuestionAnswer = (value: string) => {
    setMainQuestionAnswer(value)
  }

  const handleSubQuestionAnswer = (subQuestionId: string, value: string) => {
    setSubQuestionAnswers(prev => ({
      ...prev,
      [subQuestionId]: value
    }))
  }

  const handleContinue = () => {
    if (!mainQuestionAnswered) {
      // Save main question answer
      if (mainQuestionAnswer.trim()) {
        const updatedQuestions = [...questions]
        updatedQuestions[currentMainQuestionIndex].answer = mainQuestionAnswer
        setQuestions(updatedQuestions)
        setMainQuestionAnswered(true)
      }
    } else {
      // Check if all sub-questions are answered
      const currentMainQ = questions[currentMainQuestionIndex]
      const allSubQuestionsAnswered = currentMainQ.subQuestions.every(sq => 
        subQuestionAnswers[sq.id] && subQuestionAnswers[sq.id].trim()
      )

      if (allSubQuestionsAnswered) {
        // Save sub-question answers
        const updatedQuestions = [...questions]
        currentMainQ.subQuestions.forEach(sq => {
          sq.answer = subQuestionAnswers[sq.id] || ''
        })
        setQuestions(updatedQuestions)

        // Move to next question or submit
        if (currentMainQuestionIndex < questions.length - 1) {
          setCurrentMainQuestionIndex(currentMainQuestionIndex + 1)
          setMainQuestionAnswered(false)
          setMainQuestionAnswer('')
          setSubQuestionAnswers({})
        } else {
          // Last question - submit all answers
          submitAnswers(updatedQuestions)
        }
      }
    }
  }

  const submitAnswers = async (allQuestions: QuestionItem[]) => {
    try {
      // Send in the same format as questions.json
      const backendRequest = {
        questions: allQuestions
      }

      // ONLY 1 API CALL - when user clicks submit
      const response = await fetch('http://127.0.0.1:8000/submit-text-answer', {
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
      
      // Print the full API response to console
      console.log('API Response:', result)
      
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

  const canContinue = () => {
    if (!mainQuestionAnswered) {
      return mainQuestionAnswer.trim().length > 0
    } else {
      const mainQ = questions[currentMainQuestionIndex]
      return mainQ.subQuestions.every(sq => 
        subQuestionAnswers[sq.id] && subQuestionAnswers[sq.id].trim()
      )
    }
  }

  const isLastQuestion = currentMainQuestionIndex === questions.length - 1

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

  const currentMainQuestion = questions[currentMainQuestionIndex]
  const totalQuestions = questions.reduce((total, q) => total + 1 + q.subQuestions.length, 0)
  const currentQuestionNumber = questions.slice(0, currentMainQuestionIndex).reduce((count, q) => 
    count + 1 + q.subQuestions.length, 0
  ) + (mainQuestionAnswered ? 1 + currentMainQuestion.subQuestions.filter(sq => 
    subQuestionAnswers[sq.id] && subQuestionAnswers[sq.id].trim()
  ).length : 1)
  const progressPercentage = (currentQuestionNumber / totalQuestions) * 100

  return (
    <div className="questionnaire-card">
      <div className="questionnaire-header">
        <h2>Personal Health Questionnaire</h2>
        <div className="progress-info">
          Question {currentQuestionNumber} of {totalQuestions}
        </div>
      </div>
      
      <div className="progress-bar">
        <div 
          className="progress-fill" 
          style={{ width: `${progressPercentage}%` }}
        />
      </div>

      <div className="questions-container">
        {/* Main Question */}
        <div className="main-question">
          <div className="question-number">{currentMainQuestionIndex + 1}</div>
          <div className="question-content">
            <p className="question-text">{currentMainQuestion.mainQuestion}</p>
            
            <div className="text-input-container">
              <textarea
                className="text-input"
                placeholder="Type your answer..."
                value={mainQuestionAnswer}
                onChange={(e) => handleMainQuestionAnswer(e.target.value)}
                rows={4}
                disabled={mainQuestionAnswered}
              />
            </div>
          </div>
        </div>

        {/* Sub-questions - shown after main question is answered */}
        {mainQuestionAnswered && (
          <div className="sub-questions-container">
            {currentMainQuestion.subQuestions.map((subQ, index) => (
              <div key={subQ.id} className="sub-question">
                <div className="question-number">
                  {currentMainQuestionIndex + 1}{String.fromCharCode(97 + index)}
                </div>
                <div className="question-content">
                  <p className="question-text">{subQ.text}</p>
                  <div className="text-input-container">
                    <textarea
                      className="text-input"
                      placeholder="Type your answer..."
                      value={subQuestionAnswers[subQ.id] || ''}
                      onChange={(e) => handleSubQuestionAnswer(subQ.id, e.target.value)}
                      rows={3}
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Continue/Submit Button */}
        <div className="continue-container">
          <button
            className="continue-btn"
            onClick={handleContinue}
            disabled={!canContinue()}
          >
            {isLastQuestion && mainQuestionAnswered ? 'Submit' : 'Continue'}
            <FiArrowRight />
          </button>
        </div>
      </div>
    </div>
  )
}

export default Questionnaire
