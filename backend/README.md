# Audio Question App

A FastAPI-based application that allows users to submit audio answers to questions with automatic transcription, and provides admin dashboard for reviewing submissions.

## Features

- **User Submissions**: Users submit audio answers with their name (UUID generated automatically)
- **Audio Transcription**: Automatic speech-to-text using OpenAI Whisper (tiny model)
- **Admin Authentication**: JWT-based admin login
- **Admin Dashboard**: View all user submissions with transcripts
- **Secure Audio Storage**: Audio files stored securely on server

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Initialize the database with sample data:**
   ```bash
   python init_db.py
   ```

3. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at `http://localhost:8000`

## API Documentation

### Authentication

Admin endpoints require JWT authentication. Obtain a token by logging in via `/auth/token`.

### Endpoints

#### 1. GET /

**Description:** Root endpoint returning API information.

**Response:**
```json
{
  "message": "Audio Question App API"
}
```

**Dummy Data:**
- Request: `GET http://localhost:8000/`
- Response: As above.

#### 2. GET /questions

**Description:** Retrieve all questions ordered by their sequence.

**Response Model:** List of QuestionResponse
```json
[
  {
    "id": 1,
    "text": "What is your name?",
    "order": 1
  },
  {
    "id": 2,
    "text": "How old are you?",
    "order": 2
  }
]
```

**Dummy Data:**
- Request: `GET http://localhost:8000/questions`
- Response:
```json
[
  {
    "id": 1,
    "text": "What is your name?",
    "order": 1
  },
  {
    "id": 2,
    "text": "How old are you?",
    "order": 2
  },
  {
    "id": 3,
    "text": "What is your favorite color?",
    "order": 3
  },
  {
    "id": 4,
    "text": "Describe your daily routine.",
    "order": 4
  },
  {
    "id": 5,
    "text": "What are your hobbies?",
    "order": 5
  }
]
```

#### 3. POST /auth/token

**Description:** Admin login to obtain JWT access token.

**Request Body:** OAuth2PasswordRequestForm (form data)
- username: string
- password: string

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Dummy Data:**
- Request: `POST http://localhost:8000/auth/token`
  - Form data: username=Admin, password=admin123
- Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3OC1hYmNkLTEyMzQtZWY5MC0xMjM0NTY3ODlhYmMiLCJleHAiOjE2MzQ1Njc4OTB9.example",
  "token_type": "bearer"
}
```

#### 4. POST /submit-answer

**Description:** Submit an audio answer to a question. Creates user if not exists, saves audio file, transcribes it, and stores the answer.

**Request:** Multipart form data
- name: string (user name)
- question_id: integer
- audio: file (audio file, e.g., .wav)

**Response:**
```json
{
  "message": "Answer submitted successfully",
  "user_uuid": "123e4567-e89b-12d3-a456-426614174000",
  "transcript": "My name is John Doe."
}
```

**Dummy Data:**
- Request: `POST http://localhost:8000/submit-answer`
  - Form data: name=John Doe, question_id=1
  - File: audio.wav (sample audio file saying "My name is John Doe")
- Response:
```json
{
  "message": "Answer submitted successfully",
  "user_uuid": "123e4567-e89b-12d3-a456-426614174000",
  "transcript": "My name is John Doe."
}
```

#### 5. GET /audio/{filename}

**Description:** Download an audio file by filename.

**Parameters:**
- filename: string (path to audio file)

**Response:** Audio file (binary)

**Dummy Data:**
- Request: `GET http://localhost:8000/audio/123e4567-e89b-12d3-a456-426614174000_1_abc123.wav`
- Response: Binary audio file data

#### 6. GET /admin/users

**Description:** List all non-admin users with their answer counts. Requires admin authentication.

**Headers:**
- Authorization: Bearer {token}

**Response Model:** List of UserSummary
```json
[
  {
    "uuid": "123e4567-e89b-12d3-a456-426614174000",
    "name": "John Doe",
    "answers_count": 3
  }
]
```

**Dummy Data:**
- Request: `GET http://localhost:8000/admin/users`
  - Header: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
- Response:
```json
[
  {
    "uuid": "123e4567-e89b-12d3-a456-426614174000",
    "name": "John Doe",
    "answers_count": 5
  },
  {
    "uuid": "456e7890-f12c-34e5-b678-901234567890",
    "name": "Jane Smith",
    "answers_count": 2
  }
]
```

#### 7. GET /admin/answers

**Description:** Get all answers with details, optionally filtered by user_uuid or question_id. Requires admin authentication.

**Query Parameters:**
- user_uuid: string (optional)
- question_id: integer (optional)

**Headers:**
- Authorization: Bearer {token}

**Response Model:** List of UserAnswers
```json
[
  {
    "user": {
      "uuid": "123e4567-e89b-12d3-a456-426614174000",
      "name": "John Doe",
      "answers_count": 3
    },
    "answers": [
      {
        "id": 1,
        "question_text": "What is your name?",
        "audio_file_path": "uploads/123e4567-e89b-12d3-a456-426614174000_1_abc123.wav",
        "transcript_text": "My name is John Doe.",
        "created_at": "2023-10-01T12:00:00"
      }
    ]
  }
]
```

**Dummy Data:**
- Request: `GET http://localhost:8000/admin/answers`
  - Header: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
- Response:
```json
[
  {
    "user": {
      "uuid": "123e4567-e89b-12d3-a456-426614174000",
      "name": "John Doe",
      "answers_count": 5
    },
    "answers": [
      {
        "id": 1,
        "question_text": "What is your name?",
        "audio_file_path": "uploads/123e4567-e89b-12d3-a456-426614174000_1_abc123.wav",
        "transcript_text": "My name is John Doe.",
        "created_at": "2023-10-01T12:00:00Z"
      },
      {
        "id": 2,
        "question_text": "How old are you?",
        "audio_file_path": "uploads/123e4567-e89b-12d3-a456-426614174000_2_def456.wav",
        "transcript_text": "I am 25 years old.",
        "created_at": "2023-10-01T12:05:00Z"
      }
    ]
  }
]
```

#### 8. POST /submit-text-answer

**Description:** Submit text-based answers to questions and get depression score prediction.

**Request Body:**
```json
{
  "questions": [
    {
      "id": 1,
      "mainQuestion": "How are you feeling today?",
      "answer": "I'm feeling okay, a bit stressed about work.",
      "subQuestions": [
        { "id": "1a", "text": "What kind of stress are you experiencing?", "answer": "Work deadlines and pressure from management." },
        { "id": "1b", "text": "How long have you been feeling this way?", "answer": "About two months now." }
      ]
    },
    {
      "id": 2,
      "mainQuestion": "How's your sleep been?",
      "answer": "Not great, I have trouble falling asleep.",
      "subQuestions": [
        { "id": "2a", "text": "Do you wake up during the night?", "answer": "Yes, I wake up around 3 AM and can't go back to sleep." }
      ]
    }
  ]
}
```

**Response:**
```json
{
  "prediction": 21.93
}
```

**Data Transformation:**
The API automatically converts the question/answer format to "turns" format:
- Takes: `mainQuestion`, `answer`, `subQuestions[].text`, `subQuestions[].answer`
- Converts to: `["question1", "answer1", "subquestion1", "subanswer1", ...]`
- Processes through depression scoring model
- Returns only the prediction score

**Logging:**
All requests are automatically logged to `user_data/user_data.jsonl` in the format:
```json
{
  "id": "0",
  "turns": ["question1", "answer1", "subquestion1", "subanswer1", ...],
  "labels": [21.93]
}
```

**Dummy Data:**
- Request: `POST http://localhost:8000/submit-text-answer`
  - Body: Sample question/answer JSON as shown above
- Response:
```json
{
  "prediction": 21.931638341297027
}
```

**Current Implementation:**
- Uses the actual `get_depression_score` function from `src/inference_service.py`
- No fallback - requires all dependencies to be installed
- Uses real ML model for depression scoring
- Logs all data to JSONL file
- Increments ID counter starting from 0

## Data Models

### User
```json
{
  "id": "string (UUID)",
  "name": "string",
  "is_admin": "boolean",
  "hashed_password": "string (nullable, for admins)",
  "created_at": "datetime"
}
```

### Question
```json
{
  "id": "integer",
  "text": "string",
  "order": "integer"
}
```

### Answer
```json
{
  "id": "integer",
  "user_uuid": "string (UUID)",
  "question_id": "integer",
  "audio_file_path": "string",
  "transcript_text": "string (nullable)",
  "created_at": "datetime"
}
```

## Default Admin Account

- Name: `Admin`
- Password: `admin123`

## Notes

- Audio files are stored in the `uploads/` directory
- Transcription uses Whisper tiny model (runs locally)
- Admin JWT tokens expire after 30 minutes
- CORS is enabled for all origins (configure for production)

## Replacing the Dummy Depression Scoring Function

The current implementation uses a dummy function for depression scoring. To replace it with the actual model:

### Step 1: Install Dependencies
```bash
pip install torch transformers sentence-transformers tqdm accelerate
```

### Step 2: Update the imports in `app/routers/answer_router.py`
Replace this line:
```python
# For now, use a dummy function to avoid dependency issues
def get_depression_score_dummy(transcript_turns, model=None, tokenizer=None, device=None, turn_batch_size=16):
    import random
    return random.uniform(5.0, 25.0)
```

With:
```python
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.inference_service import load_artifacts, get_depression_score, set_device
```

### Step 3: Update the initialize_model() function
Replace the dummy implementation with:
```python
def initialize_model():
    """Initialize the depression scoring model and tokenizer."""
    global _model, _tokenizer, _device
    
    if _model is not None:
        return  # Already initialized
    
    try:
        print("Initializing depression scoring model...")
        _device = set_device()
        
        # Update this path to your actual model location
        model_path = "/path/to/your/model_2_15.pt"
        
        _model, _tokenizer = load_artifacts(
            model_path=model_path,
            tokenizer_name="sentence-transformers/all-distilroberta-v1",
            device=_device
        )
        print("Depression scoring model initialized successfully")
      
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        _model = None
        _tokenizer = None
        _device = None
```

### Step 4: Update the API endpoint
Replace:
```python
# Get depression score (using dummy function for testing)
score = get_depression_score_dummy(transcript_turns, model=_model, tokenizer=_tokenizer, device=_device, turn_batch_size=16)
```

With:
```python
# Get depression score
score = get_depression_score(
    transcript_turns=turns,
    model=_model,
    tokenizer=_tokenizer,
    device=_device,
    turn_batch_size=16
)
```

### Step 5: Update model path
Update the model path in `initialize_model()` to point to your actual trained model file.

The rest of the implementation will work seamlessly with the real model.