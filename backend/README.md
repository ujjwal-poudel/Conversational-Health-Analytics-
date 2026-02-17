# Conversational Health Analytics — Backend

The backend for the Conversational Health Analytics platform. It orchestrates a voice-and-text-based clinical interview to screen for depression, leveraging a multi-modal AI pipeline.

## Features

- **Multi-modal Conversations**: Supports both text-based chat and real-time audio interaction.
- **LLM-driven Dialogue**: Uses **Groq (Llama 3)** to paraphrase questions and generate empathetic responses.
- **Clinical Screening**: Implements the **PHQ-8** depression screening protocol.
- **Speech Technologies**:
  - **STT**: OpenAI Whisper (local inference) for user speech transcription.
  - **TTS**: Piper TTS (local neural TTS) for bot voice generation.
- **Depression Scoring**: A fine-tuned **RoBERTa** model predicts depression severity from conversation transcripts.
- **Data Persistence**: Stores all session data and scores in **Firebase Firestore**.

## Tech Stack

- **Framework**: FastAPI (Python 3.10+)
- **Inference**: PyTorch, Transformers, ONNX Runtime
- **Database**: Firebase Admin SDK (Firestore)
- **External APIs**: Groq API (for LLM generation)

## Setup & Local Development

### 1. Prerequisites
- Python 3.10+
- `ffmpeg` (required for audio processing)
- A Groq API key
- A Firebase service account JSON file

### 2. Installation

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the `backend/` directory (use `.env.example` as a template):

```bash
cp .env.example .env
nano .env
```

**Required variables:**
```ini
GROQ_API_KEY=your_groq_api_key
FIREBASE_CREDENTIALS_PATH=path/to/firebase_credentials.json
# Optional model paths (defaults provided in code)
# ROBERTA_MODEL_PATH=models/roberta/...
# PIPER_MODEL_PATH=models/piper/...
```

### 4. Running the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`.

## Documentation

- **[API Documentation](API_DOCUMENTATION.md)**: Detailed reference for all endpoints (Text Chat, Audio Chat, Schemas).
- **[EC2 Deployment Guide](EC2_DEPLOYMENT.md)**: Step-by-step instructions for deploying to AWS EC2 using Docker.

## Project Structure

```
backend/
├── app/
│   ├── api/            # API Routers (v1/chat, v1/audio)
│   ├── conversation/   # Core engine, dialogue management, LLM integration
│   ├── audio/          # STT (Whisper) and TTS (Piper) services
│   ├── schemas/        # Pydantic models
│   └── services/       # Firebase integration
├── src/                # ML Models (Inference, Fusion, HDSC architecture)
├── models/             # Local model weights (RoBERTa, Piper, etc.)
├── main.py             # Application entry point
└── requirements.txt    # Python dependencies
```