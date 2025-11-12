# Mental Health App - Frontend

A React TypeScript frontend application for conversational health analytics, designed to match the Figma specifications exactly.

## Features

- **Dual Mode Input**: Toggle between Text and Audio modes for answering questions
- **Text Mode**: Type answers with a clean, intuitive interface
- **Audio Mode**: Record audio responses with built-in playback and re-recording capabilities
- **Progress Tracking**: Visual progress bar showing questionnaire completion
- **Results Visualization**: Depression severity index displayed with an interactive gauge chart
- **Responsive Design**: Clean, modern UI matching the provided Figma designs

## Technology Stack

- **React 18** with **TypeScript**
- **Vite** for fast development and builds
- **React Icons** for consistent iconography
- **Axios** for API communication
- **Native Web APIs** for audio recording (MediaRecorder API)

## Setup Instructions

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

### Running the Development Server

Start the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Building for Production

Build the production-ready application:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Layout.tsx           # Main layout with nav and footer
│   │   ├── Layout.css
│   │   ├── Questionnaire.tsx    # Main questionnaire component
│   │   ├── Questionnaire.css
│   │   ├── AudioRecorder.tsx    # Audio recording component
│   │   ├── AudioRecorder.css
│   │   ├── Results.tsx          # Results page with gauge
│   │   └── Results.css
│   ├── App.tsx                  # Main app component
│   ├── App.css
│   ├── main.tsx                 # Entry point
│   └── index.css                # Global styles
├── index.html
├── package.json
├── tsconfig.json
├── vite.config.ts
└── README.md
```

## API Integration

The frontend is configured to proxy API requests to the FastAPI backend running on `http://localhost:8000`.

### API Endpoints Used

- `GET /api/questions` - Fetch all questions
- `POST /api/submit-answer` - Submit text or audio answers
- `GET /api/audio/{filename}` - Retrieve audio files

## Design Specifications

The frontend matches the Figma designs with:

- **Primary Color**: `#613eff` (purple for active states, buttons)
- **Background**: `#f5f5f5` (light grey)
- **Card Background**: `#ffffff` (white)
- **Text Colors**: `#333` (primary), `#666` (secondary), `#999` (tertiary)
- **Border Radius**: `12px` (cards), `6px` (buttons), `8px` (inputs)
- **Font**: System fonts for optimal readability

## Features in Detail

### Text Mode
- Clean textarea for typing responses
- Character counter (optional)
- Continue button activates when text is entered
- Previous answers displayed above in grey/disabled state

### Audio Mode
- Record button with microphone icon
- Recording indicator with timer
- Stop and cancel options during recording
- Playback controls after recording
- Re-record functionality
- Visual waveform placeholder

### Results Page
- Full-width progress bar (100%)
- Circular gauge showing depression severity score (0-27 scale)
- Color-coded severity levels
- Recommendation message based on score
- Severity scale explanation
- Medical disclaimer

## Browser Compatibility

- Chrome/Edge (recommended for audio recording)
- Firefox
- Safari (requires HTTPS for audio recording)

## Notes

- Audio recording requires microphone permissions
- The app uses the MediaRecorder API (not supported in older browsers)
- For production deployment, ensure HTTPS is enabled for audio features

## License

Part of the AI Capstone Project - Conversational Health Analytics

