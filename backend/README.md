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