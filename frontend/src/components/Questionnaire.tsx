import { useState, useEffect } from 'react'
import { FiArrowRight, FiMic } from 'react-icons/fi'
import './Questionnaire.css'
import AudioRecorder from './AudioRecorder'

interface Question {
  id: number
  text: string
  order: number
}

interface Answer {
  questionId: number
  text?: string
  audioUrl?: string
  audioDuration?: number
}

interface QuestionnaireProps {
  onComplete: (score: number) => void
}

const Questionnaire: React.FC<QuestionnaireProps> = ({ onComplete }) => {
  const [mode, setMode] = useState<'text' | 'audio'>('text')
  const [questions, setQuestions] = useState<Question[]>([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [answers, setAnswers] = useState<Answer[]>([])
  const [textAnswer, setTextAnswer] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Fetch questions from backend
    fetchQuestions()
  }, [])

  const fetchQuestions = async () => {
    try {
      const response = await fetch('/api/questions')
      const data = await response.json()
      setQuestions(data)
      setLoading(false)
    } catch (error) {
      console.error('Error fetching questions:', error)
      // Fallback to default questions
      setQuestions([
        { id: 1, text: "Can you tell me about things you used to enjoy — hobbies, outings, or small pleasures — and whether you still find them enjoyable?", order: 1 },
        { id: 2, text: "When did you first notice a change?", order: 2 },
        { id: 3, text: "How has your sleep been lately?", order: 3 },
        { id: 4, text: "Have you noticed any changes in your appetite or weight?", order: 4 },
        { id: 5, text: "How would you describe your energy levels?", order: 5 }
      ])
      setLoading(false)
    }
  }

  const handleTextContinue = () => {
    if (textAnswer.trim()) {
      const newAnswer: Answer = {
        questionId: questions[currentQuestionIndex].id,
        text: textAnswer
      }
      
      setAnswers([...answers, newAnswer])
      
      if (currentQuestionIndex < questions.length - 1) {
        setCurrentQuestionIndex(currentQuestionIndex + 1)
        setTextAnswer('')
      } else {
        // All questions answered, submit and show results
        submitAnswers([...answers, newAnswer])
      }
    }
  }

  const handleAudioComplete = (audioBlob: Blob, duration: number) => {
    const audioUrl = URL.createObjectURL(audioBlob)
    const newAnswer: Answer = {
      questionId: questions[currentQuestionIndex].id,
      audioUrl: audioUrl,
      audioDuration: duration
    }
    
    setAnswers([...answers, newAnswer])
    
    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1)
    } else {
      // All questions answered, submit and show results
      submitAnswers([...answers, newAnswer])
    }
  }

  const submitAnswers = async (allAnswers: Answer[]) => {
    // Here you would send answers to backend
    // For now, calculate a mock depression score
    const mockScore = Math.floor(Math.random() * 20) + 10 // Random score between 10-30
    
    // Simulate API delay
    setTimeout(() => {
      onComplete(mockScore)
    }, 500)
  }

  const progressPercentage = questions.length > 0 
    ? ((currentQuestionIndex + 1) / questions.length) * 100 
    : 0

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

  const currentQuestion = questions[currentQuestionIndex]

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
        {/* Show previous answers */}
        {answers.map((answer, index) => (
          <div key={index} className="answered-question">
            <div className="question-number">{index + 1}</div>
            <div className="question-content">
              <p className="question-text">{questions[index].text}</p>
              {answer.text && (
                <div className="answer-display">
                  <p>{answer.text}</p>
                </div>
              )}
              {answer.audioUrl && (
                <div className="audio-display">
                  <audio controls src={answer.audioUrl} />
                  <span className="audio-duration">
                    {Math.floor(answer.audioDuration || 0)}s
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Current question */}
        <div className="current-question">
          <div className="question-number active">{currentQuestionIndex + 1}</div>
          <div className="question-content">
            <p className="question-text">{currentQuestion.text}</p>
            
            {mode === 'text' ? (
              <div className="text-input-container">
                <textarea
                  className="text-input"
                  placeholder="Type your answer..."
                  value={textAnswer}
                  onChange={(e) => setTextAnswer(e.target.value)}
                  rows={4}
                />
                <button
                  className="continue-btn"
                  onClick={handleTextContinue}
                  disabled={!textAnswer.trim()}
                >
                  {currentQuestionIndex < questions.length - 1 ? 'Continue' : 'Submit'}
                  <FiArrowRight />
                </button>
              </div>
            ) : (
              <AudioRecorder 
                key={currentQuestionIndex} 
                onComplete={handleAudioComplete} 
              />
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Questionnaire

