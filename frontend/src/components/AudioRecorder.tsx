import { useState, useRef, useEffect } from 'react'
import { FiMic, FiSquare, FiPlay, FiPause, FiRotateCcw } from 'react-icons/fi'
import './AudioRecorder.css'

interface AudioRecorderProps {
  onComplete: (audioBlob: Blob, duration: number) => void
}

const AudioRecorder: React.FC<AudioRecorderProps> = ({ onComplete }) => {
  const [isRecording, setIsRecording] = useState(false)
  const [hasRecording, setHasRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string>('')
  const [isPlaying, setIsPlaying] = useState(false)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl)
      }
    }
  }, [audioUrl])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        const url = URL.createObjectURL(blob)
        setAudioBlob(blob)
        setAudioUrl(url)
        setHasRecording(true)
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
      setRecordingTime(0)
      
      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1)
      }, 1000)
    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert('Could not access microphone. Please grant permission.')
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }

  const cancelRecording = () => {
    if (isRecording) {
      stopRecording()
    }
    
    setIsRecording(false)
    setHasRecording(false)
    setRecordingTime(0)
    setAudioBlob(null)
    
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl)
      setAudioUrl('')
    }
    
    if (timerRef.current) {
      clearInterval(timerRef.current)
    }
  }

  const recordAgain = () => {
    cancelRecording()
  }

  const handleContinue = () => {
    if (audioBlob) {
      onComplete(audioBlob, recordingTime)
    }
  }

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
  }

  if (hasRecording) {
    return (
      <div className="audio-recorder">
        <div className="audio-playback">
          <audio
            ref={audioRef}
            src={audioUrl}
            onEnded={() => setIsPlaying(false)}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
          <button className="playback-btn" onClick={togglePlayback}>
            {isPlaying ? <FiPause size={20} /> : <FiPlay size={20} />}
          </button>
          <div className="playback-info">
            <div className="waveform-placeholder"></div>
            <span className="recording-time">{formatTime(recordingTime)}</span>
          </div>
        </div>
        <div className="recording-actions">
          <button className="record-again-btn" onClick={recordAgain}>
            <FiRotateCcw />
            Record again
          </button>
          <button className="continue-btn" onClick={handleContinue}>
            Continue
            <FiMic />
          </button>
        </div>
      </div>
    )
  }

  if (isRecording) {
    return (
      <div className="audio-recorder">
        <div className="recording-indicator">
          <div className="recording-pulse"></div>
          <span>Recording...</span>
          <span className="recording-time">{formatTime(recordingTime)}</span>
        </div>
        <div className="recording-actions">
          <button className="cancel-btn" onClick={cancelRecording}>
            Cancel
          </button>
          <button className="stop-recording-btn" onClick={stopRecording}>
            <FiSquare />
            Stop recording
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="audio-recorder">
      <button className="record-btn" onClick={startRecording}>
        <FiMic size={20} />
        Record
      </button>
    </div>
  )
}

export default AudioRecorder

