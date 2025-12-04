import { useState } from 'react'
import Layout from './components/Layout'
import Chatbot from './components/Chatbot'
import AudioChat from './components/AudioChat'
import './App.css'

function App() {
  const [view, setView] = useState<'chatbot' | 'audiochat'>('chatbot')

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
          💬 Text Chat
        </button>
        <button
          onClick={() => setView('audiochat')}
          style={{
            padding: '8px 16px',
            background: view === 'audiochat' ? '#764ba2' : 'rgba(255,255,255,0.1)',
            border: 'none',
            borderRadius: '20px',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          🎤 Voice Chat
        </button>
      </div>

      {view === 'chatbot' ? <Chatbot /> : <AudioChat />}
    </Layout>
  )
}

export default App

