import { useEffect, useMemo, useRef, useState } from "react";
import mermaid from "mermaid";

/**
 * AboutPage.jsx
 *
 * Install Mermaid:
 *   npm i mermaid
 *
 * Notes:
 * - This component renders Mermaid diagrams client-side.
 * - If you use Next.js SSR, keep Mermaid rendering inside useEffect (already done).
 */

interface MermaidBlockProps {
    title: string;
    caption?: string;
    code: string;
}

function MermaidBlock({ title, caption, code }: MermaidBlockProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [svg, setSvg] = useState<string>("");

    useEffect(() => {
        let cancelled = false;

        async function render() {
            try {
                // Unique ID per render to avoid collisions
                const id = `mmd-${Math.random().toString(36).slice(2)}`;

                // Render returns SVG text directly
                const { svg } = await mermaid.render(id, code);

                if (!cancelled) {
                    setSvg(svg);
                }
            } catch (e: any) {
                console.error('Mermaid render error:', e);
                // Silent fail — don't render raw error text to the user
                if (!cancelled) {
                    setSvg(`<p style="color:rgba(255,255,255,.45);font-size:13px;padding:8px">Diagram unavailable</p>`);
                }
            }
        }

        render();
        return () => {
            cancelled = true;
        };
    }, [code]);

    return (
        <div className="card">
            <div className="cardHead">
                <h3>{title}</h3>
                {caption ? <p className="muted">{caption}</p> : null}
            </div>

            <div className="diagramShell" ref={containerRef}>
                <div
                    className="mermaidOut"
                    // Mermaid returns raw SVG string
                    dangerouslySetInnerHTML={{ __html: svg }}
                />
            </div>
        </div>
    );
}

function AnimatedArchitecture() {
    return (
        <div className="card">
            <div className="cardHead">
                <h3>Animated Architecture (Live Flow)</h3>
                <p className="muted">
                    Two parallel pipelines: Text Chat (RoBERTa) and Audio Chat (Whisper → Wav2Vec2 → Lasso), fused via min-score strategy.
                </p>
            </div>

            <div className="animWrap" aria-label="Animated architecture flow">
                {/* Text Chat row */}
                <p style={{ fontSize: 12, color: 'rgba(255,255,255,.45)', marginBottom: 8, marginTop: 0 }}>TEXT MODE</p>
                <div className="animRow" style={{ marginBottom: 18 }}>
                    <div className="animNode glow1">
                        <b>User</b>
                        <small>Types response</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow2">
                        <b>React UI</b>
                        <small>Frontend</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow3">
                        <b>FastAPI</b>
                        <small>Rate limited</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow2">
                        <b>Conv Engine</b>
                        <small>Groq LLM rewriter</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow1">
                        <b>RoBERTa</b>
                        <small>Text inference</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow3">
                        <b>Fusion</b>
                        <small>min strategy</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow1">
                        <b>Firebase</b>
                        <small>Store result</small>
                    </div>
                </div>

                {/* Audio Chat row */}
                <p style={{ fontSize: 12, color: 'rgba(255,255,255,.45)', marginBottom: 8 }}>AUDIO MODE</p>
                <div className="animRow">
                    <div className="animNode glow1">
                        <b>User</b>
                        <small>Speaks (WebM)</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow2">
                        <b>Whisper STT</b>
                        <small>faster-whisper</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow3">
                        <b>Preprocessor</b>
                        <small>16kHz + 300Hz HPF</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow2">
                        <b>Wav2Vec2</b>
                        <small>768-dim embed</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow1">
                        <b>PCA + Lasso</b>
                        <small>768→200 + predict</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow3">
                        <b>Fusion</b>
                        <small>min strategy</small>
                    </div>
                    <div className="animLink"><span className="dotRun" /></div>
                    <div className="animNode glow1">
                        <b>Firebase</b>
                        <small>Store result</small>
                    </div>
                </div>

                <div className="animLegend">
                    <span className="pill">Whisper STT transcription</span>
                    <span className="pill">Wav2Vec2 + Prosody features</span>
                    <span className="pill">RoBERTa text model</span>
                    <span className="pill">Min-fusion scoring</span>
                    <span className="pill">Firebase Firestore</span>
                </div>
            </div>
        </div>
    );
}

export default function AboutPage() {
    useEffect(() => {
        mermaid.initialize({
            startOnLoad: false,
            securityLevel: "loose",
            theme: "dark",
            darkMode: true,
            fontFamily:
                'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial',
            // Mermaid "dark" theme variables
            themeVariables: {
                background: "transparent",
                primaryColor: "rgba(255,255,255,0.06)",
                primaryTextColor: "rgba(255,255,255,0.90)",
                primaryBorderColor: "rgba(255,255,255,0.14)",
                lineColor: "rgba(255,255,255,0.25)",
                secondaryColor: "rgba(139,92,246,0.22)",
                tertiaryColor: "rgba(34,211,238,0.16)",
                noteBkgColor: "rgba(251,191,36,0.10)",
                noteTextColor: "rgba(255,255,255,0.82)",
                edgeLabelBackground: "rgba(0,0,0,0.55)",
            },
        });
    }, []);

    const archMermaid = useMemo(
        () => `
flowchart TB
  U(["User"]) --> FE["React Frontend"]
  FE --> RL["Rate Limiter: slowapi"]
  RL --> CE["Conversation Engine"]
  CE --> LLM["Groq LLM Rewriter"]
  CE --> VD["Vague Detector"]
  CE --> SC["Sufficiency Checker"]
  CE --> QA["Q/A Recorder"]

  FE -->|"Audio mode"| STT["Whisper STT: faster-whisper"]
  STT --> CE

  QA --> TXT["Text Inference: RoBERTa distilroberta-v1"]
  QA --> AUD["Audio Pipeline"]
  AUD --> PRE["Preprocessor: resample 16kHz + high-pass 300Hz"]
  PRE --> W2V["Wav2Vec2: 768-dim embeddings"]
  PRE --> PRO["Prosody: F0 + RMS + ZCR + spectral 13-dim"]
  W2V --> PCA["PCA: 768 to 200 dims"]
  PCA --> LAS["Lasso: scaler + selector + predict"]
  PRO --> LAS

  TXT --> FUS["Fusion Service: min strategy"]
  LAS --> FUS
  FUS --> FB["Firebase Firestore"]
  FUS --> FE

  classDef ui fill:#8B5CF62E,stroke:#FFFFFF29,color:#fff;
  classDef api fill:#22D3EE24,stroke:#FFFFFF29,color:#fff;
  classDef ml fill:#34D3991A,stroke:#FFFFFF29,color:#fff;
  classDef storage fill:#FBBF241F,stroke:#FFFFFF29,color:#fff;
  class U,FE ui;
  class RL,CE,LLM,VD,SC,QA,STT api;
  class TXT,AUD,PRE,W2V,PRO,PCA,LAS,FUS ml;
  class FB storage;
`,
        []
    );

    const backendSeq = useMemo(
        () => `
sequenceDiagram
  autonumber
  participant User
  participant FE as Frontend React
  participant RL as Rate Limiter
  participant API as FastAPI Backend
  participant CE as Conv Engine
  participant LLM as Groq LLM Rewriter
  participant STT as Whisper STT
  participant ML as RoBERTa / Lasso
  participant FB as Firebase Firestore

  Note over User,FE: Text Chat Mode
  User->>FE: Start conversation
  FE->>RL: POST /api/v1/chat/start
  RL->>API: 10 req/min allowed
  API->>CE: Create session + ConversationEngine
  CE->>LLM: Generate first question
  LLM-->>CE: Rewritten prompt
  API-->>FE: session_id + first message

  loop Each conversation turn
    User->>FE: Types answer
    FE->>RL: POST /api/v1/chat/chat
    RL->>API: 20 req/min allowed
    API->>CE: engine.process(user_message)
    CE->>LLM: Rephrase / follow-up via Groq
    LLM-->>CE: Next question
    API-->>FE: bot response + is_finished
  end

  Note over API,FB: Conversation complete
  API->>ML: RoBERTa inference on Q/A pairs
  ML-->>API: PHQ depression score
  API->>FB: save_conversation (mode=text)
  API-->>FE: depression_score + score_source

  Note over User,FE: Audio Chat Mode
  User->>FE: Speaks (WebM audio)
  FE->>RL: POST /api/v1/audio/chat/turn
  RL->>API: 15 req/min allowed
  API->>STT: Whisper transcribe audio
  STT-->>API: transcript + word timestamps
  API->>CE: engine.process(transcript)
  CE-->>API: next bot question
  API-->>FE: bot text + TTS audio paths

  Note over API,FB: Audio session complete
  API->>ML: RoBERTa text-only scoring (audio fallback)
  ML-->>API: PHQ score
  API->>FB: save_conversation (mode=audio)
  API-->>FE: depression_score + score_source
`,
        []
    );

    const mlPipeline = useMemo(
        () => `
flowchart TB
  A["DAIC-WOZ Dataset: Audio + Transcripts + PHQ-8"] --> B["Preprocessing: clean + align labels"]
  B --> C["Feature Engineering: MFCC, Delta + stats"]
  C --> D[Scaling / Normalization]
  D --> E[Train / Validation Split]
  E --> F["Model Training: LR, SVM, RF, XGB, LGBM"]
  F --> G["Tuning: Grid / Random / Optuna"]
  G --> H["Evaluation: MAE, RMSE, R-squared"]
  H --> I["Deployment Package: Saved pipeline + API"]

  classDef data fill:#FBBF241F,stroke:#FFFFFF29,color:#fff;
  classDef proc fill:#8B5CF626,stroke:#FFFFFF29,color:#fff;
  classDef train fill:#22D3EE1F,stroke:#FFFFFF29,color:#fff;
  classDef eval fill:#34D3991A,stroke:#FFFFFF29,color:#fff;
  class A data;
  class B,C,D,E proc;
  class F,G train;
  class H,I eval;
`,
        []
    );

    return (
        <div className="wrap">
            {/* Topbar */}
            <header className="topbar">
                <div className="brand" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    <button
                        onClick={() => window.location.href = '/'}
                        className="btn"
                        style={{ marginRight: '1rem', padding: '8px 12px', background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}
                    >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '4px' }}>
                            <path d="M3 9.5L12 3l9 6.5V20a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9.5z" />
                            <polyline points="9 21 9 12 15 12 15 21" />
                        </svg>
                        Home
                    </button>
                    <div className="logo" aria-hidden="true" />
                    <div className="brandText">
                        <div className="brandTitle">Conversational Health Analytics</div>
                        <div className="brandSub">About the project</div>
                    </div>
                </div>

                <nav className="nav">
                    <a className="chip" href="#overview">
                        Overview
                    </a>
                    <a className="chip" href="#data">
                        Data
                    </a>
                    <a className="chip" href="#architecture">
                        Architecture
                    </a>
                    <a className="chip" href="#models">
                        AI Models
                    </a>
                    <a className="chip" href="#pipeline">
                        Pipeline
                    </a>
                    <a className="chip" href="#tech">
                        Tech Stack
                    </a>
                    <a className="chip" href="#ethics">
                        Ethics
                    </a>
                    <a className="chip" href="#authors">
                        Authors
                    </a>
                </nav>
            </header>

            {/* Hero */}
            <section className="hero" id="overview">
                <div className="kicker">
                    <span className="pulse" /> Research Prototype • Multi-Modal AI
                </div>

                <h1>
                    Understand conversation + voice signals
                    <br />
                    with a single end-to-end AI system.
                </h1>

                <p className="heroP">
                    Conversational Health Analytics combines structured conversational prompting, audio signal processing, and
                    machine learning inference to explore early mental health signal detection from speech and language patterns.
                </p>

                <div className="callouts">
                    <span className="badge">
                        <span className="dot" /> Conversational flow
                    </span>
                    <span className="badge">
                        <span className="dot" /> Audio DSP (Librosa)
                    </span>
                    <span className="badge">
                        <span className="dot" /> ML scoring engine
                    </span>
                    <span className="badge">
                        <span className="dot" /> API-first design
                    </span>
                </div>

                <div className="heroGrid">
                    <div className="card">
                        <div className="cardHead">
                            <h3>What this project does</h3>
                            <p className="muted">
                                Guides users through topic-based prompts, extracts speech features, runs ML inference, and returns
                                structured scores as a research exploration tool.
                            </p>
                        </div>

                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Topic-driven conversation (primary/secondary prompts)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Audio preprocessing + MFCC/Δ/ΔΔ feature engineering</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Model inference (classification/regression depending on task)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Backend API built for frontend integration</span>
                            </li>
                        </ul>
                    </div>

                    <div className="notice">
                        <strong>Important notice</strong>
                        <p>
                            This system is <b>not</b> a medical diagnostic tool and does not replace professional clinical assessment.
                            It is intended for research and educational purposes only.
                        </p>
                    </div>
                </div>
            </section>

            {/* Problem & Solution */}
            <section className="section" id="problem">
                <div className="sectionHeader">
                    <h2>Problem & Proposed Solution</h2>
                    <span>Why it exists + what it aims to explore</span>
                </div>

                <div className="grid2">
                    <div className="card">
                        <div className="cardHead">
                            <h3>Problem Statement</h3>
                            <p className="muted">
                                Mental health assessment can require structured interviews and expert interpretation. Early signals are
                                subtle, access to support can be limited, and scalable approaches are valuable for research.
                            </p>
                        </div>
                    </div>

                    <div className="card">
                        <div className="cardHead">
                            <h3>Solution Approach</h3>
                            <p className="muted">
                                Combine structured conversational prompting with audio feature extraction and machine learning inference,
                                delivered through a modular backend API and web UI.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* Data */}
            <section className="section" id="data">
                <div className="sectionHeader">
                    <h2>Data Source</h2>
                    <span>Where the training/evaluation data comes from</span>
                </div>

                <div className="card">
                    <div className="cardHead">
                        <h3>DAIC-WOZ (Distress Analysis Interview Corpus – Wizard of Oz)</h3>
                        <p className="muted">
                            Used for research on depression detection from interviews. Includes participant audio, transcripts, and
                            PHQ-8 labels. Provided by USC Institute for Creative Technologies (access subject to license).
                        </p>
                    </div>

                    <ul className="list">
                        <li className="li">
                            <span className="tick">✓</span>
                            <span>Clinical-style interview conversations</span>
                        </li>
                        <li className="li">
                            <span className="tick">✓</span>
                            <span>Audio recordings + transcripts</span>
                        </li>
                        <li className="li">
                            <span className="tick">✓</span>
                            <span>PHQ-8 depression severity scores (research labels)</span>
                        </li>
                        <li className="li">
                            <span className="tick">✓</span>
                            <span>Citation: Gratch et al., LREC 2014</span>
                        </li>
                    </ul>
                </div>
            </section>

            {/* Architecture */}
            <section className="section" id="architecture">
                <div className="sectionHeader">
                    <h2>System Architecture</h2>
                    <span>Mermaid diagrams + animated flow</span>
                </div>

                <div className="grid2">
                    <MermaidBlock
                        title="High-Level Architecture"
                        caption="Flowchart overview of user → frontend → backend → DSP → ML → scoring."
                        code={archMermaid}
                    />
                    <AnimatedArchitecture />
                </div>

                <div className="stack" style={{ marginTop: 14 }}>
                    <MermaidBlock
                        title="Backend Request Lifecycle"
                        caption="Sequence diagram showing how a single request is processed end-to-end."
                        code={backendSeq}
                    />
                </div>
            </section>

            {/* Models */}
            <section className="section" id="models">
                <div className="sectionHeader">
                    <h2>AI Models Used</h2>
                    <span>Conversational layer · Speech · Audio ML · Text ML · Fusion</span>
                </div>

                <div className="grid2">
                    {/* Conversational AI */}
                    <div className="card">
                        <div className="cardHead">
                            <h3>Conversational AI Layer</h3>
                            <p className="muted">
                                Topic-driven conversation engine powered by Groq LLM for natural rephrasing,
                                with vague-response detection and sufficiency gating per topic.
                            </p>
                        </div>
                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span><b>Groq LLM</b> — llama-3.1-8b-instant for response rewriting</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Topics: mood, sleep, energy, daily routines, functioning</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Primary / secondary prompt strategy per topic</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Vague response detector + sufficiency checker</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Q/A recorder for inference pairs at session end</span>
                            </li>
                        </ul>
                    </div>

                    {/* Speech Transcription */}
                    <div className="card">
                        <div className="cardHead">
                            <h3>Speech Transcription (Audio Mode)</h3>
                            <p className="muted">
                                Audio mode uses faster-whisper to transcribe user speech (WebM) to text
                                before passing it into the conversation engine.
                            </p>
                        </div>
                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span><b>faster-whisper</b> — OpenAI Whisper, int8 quantized CPU inference</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Converts WebM → WAV via pydub before transcription</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Word-level timestamps captured per segment</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Language auto-detection included in output metadata</span>
                            </li>
                        </ul>
                    </div>
                </div>

                <div className="grid2" style={{ marginTop: 14 }}>
                    {/* Audio ML */}
                    <div className="card">
                        <div className="cardHead">
                            <h3>Audio ML Pipeline</h3>
                            <p className="muted">
                                Deep speech embeddings + classical prosody features, reduced via PCA and
                                scored with a trained Lasso regression model.
                            </p>
                        </div>
                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Preprocess: resample to 16kHz + 300Hz Butterworth high-pass filter</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span><b>Wav2Vec2</b> — superb/wav2vec2-base-superb-er, 768-dim embeddings</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span><b>Prosody</b> — F0, RMS, ZCR, spectral centroid/bandwidth/rolloff/contrast (13-dim)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span><b>PCA</b> — 768 → 200 dims + 3-segment pooling (mean/std/min/max)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span><b>Lasso</b> — scaler → feature selector → Lasso regression → PHQ score</span>
                            </li>
                        </ul>
                    </div>

                    {/* Text ML */}
                    <div className="card">
                        <div className="cardHead">
                            <h3>Text Inference Model</h3>
                            <p className="muted">
                                Fine-tuned RoBERTa model that scores the full conversation Q/A transcript
                                to produce a PHQ depression severity prediction.
                            </p>
                        </div>
                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span><b>RoBERTa</b> — sentence-transformers/all-distilroberta-v1 backbone</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Multi-class attention model fine-tuned on PHQ-8 labels</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Runs on MPS (Apple Silicon) / CUDA / CPU automatically</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Inference on Q/A turn pairs (batch size 16)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Score clamped to valid PHQ-8 range [0–24]</span>
                            </li>
                        </ul>
                    </div>
                </div>

                <div className="card" style={{ marginTop: 14 }}>
                    <div className="cardHead">
                        <h3>Deployment & Infrastructure</h3>
                        <p className="muted">
                            Backend containerized with Docker on AWS EC2. Frontend hosted on Render. Firebase for session persistence.
                        </p>
                    </div>
                    <div className="pillGrid">
                        <div className="pill">
                            AWS EC2 <span>backend host</span>
                        </div>
                        <div className="pill">
                            Docker <span>Python 3.11 slim</span>
                        </div>
                        <div className="pill">
                            Uvicorn <span>1 worker, port 8000</span>
                        </div>
                        <div className="pill">
                            Multi-stage build <span>CPU-only PyTorch</span>
                        </div>
                        <div className="pill">
                            Render <span>frontend hosting</span>
                        </div>
                        <div className="pill">
                            Firebase Firestore <span>session storage</span>
                        </div>
                        <div className="pill">
                            slowapi <span>rate limiting</span>
                        </div>
                        <div className="pill">
                            HF cache <span>volume-mounted models</span>
                        </div>
                        <div className="pill">
                            /health endpoint <span>30s Docker healthcheck</span>
                        </div>
                        <div className="pill">
                            Session reaper <span>1hr TTL, 5min interval</span>
                        </div>
                        <div className="pill">
                            CORS <span>configured for Render URL</span>
                        </div>
                        <div className="pill">
                            Non-root user <span>appuser UID 1000</span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Pipeline */}
            <section className="section" id="pipeline">
                <div className="sectionHeader">
                    <h2>Machine Learning Pipeline</h2>
                    <span>Mermaid view of end-to-end training</span>
                </div>

                <MermaidBlock
                    title="Training & Deployment Pipeline"
                    caption="Data → features → training → tuning → evaluation → deployment package."
                    code={mlPipeline}
                />
            </section>

            {/* Tech Stack */}
            <section className="section" id="tech">
                <div className="sectionHeader">
                    <h2>Tech Stack</h2>
                    <span>Frontend • Backend • ML • Deployment</span>
                </div>

                <div className="grid2">
                    <div className="card">
                        <div className="cardHead">
                            <h3>Frontend</h3>
                            <p className="muted">User-facing UI for conversation + results.</p>
                        </div>
                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>React</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Responsive layout (CSS)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>API integration</span>
                            </li>
                        </ul>
                    </div>

                    <div className="card">
                        <div className="cardHead">
                            <h3>Backend + ML</h3>
                            <p className="muted">Inference pipeline and routing.</p>
                        </div>
                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>FastAPI (Python)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Librosa (DSP)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Scikit-learn / XGBoost / LightGBM</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Optuna (tuning), NumPy, Pandas</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </section>

            {/* Ethics */}
            <section className="section" id="ethics">
                <div className="sectionHeader">
                    <h2>Ethics, Privacy & Limitations</h2>
                    <span>Responsible AI in health domains</span>
                </div>

                <div className="grid2">
                    <div className="card">
                        <div className="cardHead">
                            <h3>Ethical Positioning</h3>
                            <p className="muted">
                                The system is a research prototype. It does not provide medical advice, diagnosis, or treatment decisions.
                            </p>
                        </div>

                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Not a diagnostic tool</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Does not replace professionals</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Focuses on transparency & limitations</span>
                            </li>
                        </ul>
                    </div>

                    <div className="card">
                        <div className="cardHead">
                            <h3>Known Limitations</h3>
                            <p className="muted">
                                Generalization and fairness depend on data distribution, audio quality, and real-world deployment context.
                            </p>
                        </div>

                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Dataset context bias (clinical interview setting)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Audio quality can affect feature reliability</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Demographic/cultural bias may exist</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Needs further validation for real-world use</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </section>

            {/* Authors */}
            <section className="section" id="authors">
                <div className="sectionHeader">
                    <h2>Authors</h2>
                    <span>Who built this</span>
                </div>

                <div className="grid2">
                    <div className="card">
                        <div className="cardHead">
                            <h3>Ujjwal Poudel</h3>
                            <p className="muted">
                                Built the ML pipeline, audio feature engineering, backend API, and conversational flow logic.
                            </p>
                        </div>

                        <div className="btnRow">
                            <a className="btn primary" href="https://www.linkedin.com/in/up1/" target="_blank" rel="noreferrer">
                                LinkedIn
                            </a>
                            <a className="btn" href="https://github.com/ujjwal-poudel" target="_blank" rel="noreferrer">
                                GitHub
                            </a>
                            <a className="btn" href="mailto:ujjwal.poudel.2003@gmail.com">
                                Email
                            </a>
                        </div>

                        <p className="tinyMuted">Replace links with your real socials.</p>
                    </div>

                    <div className="card">
                        <div className="cardHead">
                            <h3>Roadmap</h3>
                            <p className="muted">Planned improvements to increase robustness and research rigor.</p>
                        </div>

                        <ul className="list">
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Real-time inference + streaming audio</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Speech foundation models (wav2vec2-style)</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Bias audits + fairness testing</span>
                            </li>
                            <li className="li">
                                <span className="tick">✓</span>
                                <span>Clinical validation collaboration</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </section>

            {/* Footer */}
            <footer className="footer">
                <p className="muted">
                    Built as a research prototype to explore AI-driven conversation + speech analytics. Interested in collaborating
                    or reviewing the architecture? Reach out via the links above.
                </p>
                <div className="btnRow">
                    <a className="btn" href="#overview">
                        Back to top
                    </a>
                    <a className="btn primary" href="#data">
                        Data source
                    </a>
                </div>
            </footer>

            {/* Styles */}
            <style>{`
        /* Make sure scroll behavior is smooth */
        html { scroll-behavior: smooth; }
        
        :root{
          --bg: var(--bg-primary);
          --panel: var(--bg-card);
          --panel2: var(--bg-secondary);
          --border: var(--border-color);
          --text: var(--text-primary);
          --muted: var(--text-secondary);

          --accent: var(--accent-purple);
          --accent2: var(--accent-teal);
          --good: #34D399; /* Keeping a distinct green for 'good/tick' states */
          --warn: #FBBF24;

          --shadow: 0 20px 60px rgba(0,0,0,.45);
          --radius: 20px;
          --radius2: 14px;
          --max: 1100px;
        }

        .wrap *{ box-sizing: border-box; }
        .wrap{
          min-height: 100vh;
          max-width: var(--max);
          margin: 0 auto;
          padding: 32px 18px 80px;
          color: var(--text);
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
          background:
            radial-gradient(1200px 600px at 20% 10%, rgba(124, 93, 250,.10), transparent 60%),
            radial-gradient(1000px 500px at 85% 15%, rgba(0, 255, 204,.08), transparent 55%),
            radial-gradient(900px 500px at 50% 90%, rgba(59, 130, 246,.08), transparent 60%),
            var(--bg);
        }

        a{ color: inherit; text-decoration: none; }
        a:hover{ text-decoration: underline; text-underline-offset: 4px; }

        .topbar{
          display:flex;
          align-items:center;
          justify-content: space-between;
          gap: 16px;
          padding: 10px 0 22px;
          flex-wrap: wrap;
        }

        .brand{
          display:flex;
          align-items:center;
          gap: 12px;
          font-weight: 800;
          letter-spacing: .3px;
        }
        .logo{
          width: 40px; height:40px;
          border-radius: 14px;
          background:
            radial-gradient(16px 16px at 30% 30%, rgba(255,255,255,.55), transparent 60%),
            linear-gradient(135deg, var(--accent), var(--accent2));
          box-shadow: 0 14px 40px rgba(124, 93, 250,.18);
          border: 1px solid rgba(255,255,255,.16);
          flex: 0 0 40px;
        }
        .brandTitle{ font-weight: 900; }
        .brandSub{ color: rgba(255,255,255,.55); font-weight: 600; font-size: 12.5px; margin-top: 2px; }

        .nav{
          display:flex;
          gap: 10px;
          flex-wrap: wrap;
          justify-content: flex-end;
        }
        .chip{
          padding: 10px 12px;
          border-radius: 999px;
          border: 1px solid var(--border);
          background: rgba(255,255,255,.04);
          color: var(--muted);
          font-size: 13px;
          transition: .2s ease;
        }
        .chip:hover{
          background: rgba(255,255,255,.14);
          color: #fff;
          text-decoration: none;
          border-color: rgba(255,255,255,.30);
          transform: translateY(-2px) scale(1.04);
          box-shadow: 0 6px 20px rgba(124,93,250,.22), 0 0 0 1px rgba(255,255,255,.08);
        }

        .hero{
          border: 1px solid var(--border);
          background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03));
          border-radius: var(--radius);
          padding: 26px 22px;
          box-shadow: var(--shadow);
          position: relative;
          overflow: hidden;
        }
        .hero:before{
          content:"";
          position:absolute;
          inset:-2px;
          background:
            radial-gradient(900px 280px at 15% 0%, rgba(124, 93, 250,.15), transparent 55%),
            radial-gradient(900px 280px at 85% 10%, rgba(0, 255, 204,.10), transparent 55%);
          pointer-events:none;
          opacity: .9;
        }
        .hero > *{ position:relative; }

        .kicker{
          display:inline-flex;
          align-items:center;
          gap:10px;
          padding: 8px 12px;
          border-radius: 999px;
          border: 1px solid rgba(255,255,255,.14);
          background: rgba(0,0,0,.25);
          color: rgba(255,255,255,.82);
          font-size: 13px;
          font-weight: 600;
        }
        .pulse{
          width:10px; height:10px;
          border-radius: 50%;
          background: var(--good);
          box-shadow: 0 0 0 0 rgba(52,211,153,.55);
          animation: pulse 1.6s infinite;
        }
        @keyframes pulse{
          0%{ box-shadow:0 0 0 0 rgba(52,211,153,.45); }
          70%{ box-shadow:0 0 0 10px rgba(52,211,153,0); }
          100%{ box-shadow:0 0 0 0 rgba(52,211,153,0); }
        }

        h1{
          margin: 12px 0 8px;
          font-size: clamp(28px, 4vw, 44px);
          line-height: 1.1;
          letter-spacing: -.7px;
        }

        .heroP{
          margin: 0;
          color: var(--muted);
          font-size: 16px;
          max-width: 72ch;
        }

        .callouts{
          display:flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 14px;
        }
        .badge{
          border: 1px solid rgba(255,255,255,.14);
          background: rgba(255,255,255,.05);
          padding: 10px 12px;
          border-radius: 999px;
          font-size: 13px;
          color: rgba(255,255,255,.78);
          display:inline-flex;
          gap:8px;
          align-items:center;
          white-space: nowrap;
        }
        .dot{
          width:8px; height:8px; border-radius: 50%;
          background: var(--accent2);
          opacity: .95;
        }

        .heroGrid{
          display:grid;
          grid-template-columns: 1.3fr .7fr;
          gap: 18px;
          margin-top: 18px;
          align-items: start;
        }

        .section{
          margin-top: 22px;
          padding: 22px;
          border: 1px solid var(--border);
          background: rgba(255,255,255,.045);
          border-radius: var(--radius);
          box-shadow: 0 10px 35px rgba(0,0,0,.25);
        }
        .sectionHeader{
          display:flex;
          justify-content: space-between;
          align-items: flex-end;
          gap: 14px;
          flex-wrap: wrap;
          margin-bottom: 14px;
        }
        .sectionHeader h2{
          margin:0;
          font-size: 20px;
          letter-spacing: -.3px;
        }
        .sectionHeader span{
          color: var(--muted);
          font-size: 13px;
        }

        .grid2{
          display:grid;
          grid-template-columns: 1fr 1fr;
          gap: 14px;
        }
        .stack{ display:block; }

        .card{
          border: 1px solid rgba(255,255,255,.11);
          background: rgba(0,0,0,.18);
          border-radius: var(--radius2);
          padding: 16px;
          overflow: hidden;
        }
        .cardHead h3{
          margin:0 0 6px;
          font-size: 15px;
          letter-spacing: -.2px;
        }
        .muted{ color: var(--muted); margin: 0; font-size: 14px; }

        .notice{
          border-radius: var(--radius2);
          border: 1px solid rgba(251,191,36,.22);
          background: rgba(251,191,36,.08);
          padding: 14px 14px;
          color: rgba(255,255,255,.86);
        }
        .notice p{ margin: 8px 0 0; color: rgba(255,255,255,.78); font-size: 13.5px; }

        .list{
          margin: 12px 0 0;
          padding: 0;
          list-style: none;
          display:grid;
          gap: 8px;
        }
        .li{
          display:flex;
          gap:10px;
          align-items:flex-start;
          color: rgba(255,255,255,.80);
          font-size: 14px;
        }
        .tick{
          width: 20px;
          height: 20px;
          border-radius: 7px;
          background: rgba(52,211,153,.14);
          border: 1px solid rgba(52,211,153,.25);
          display:flex;
          align-items:center;
          justify-content:center;
          flex: 0 0 20px;
          margin-top: 2px;
          font-size: 12px;
          color: rgba(255,255,255,.85);
        }

        /* Mermaid wrapper */
        .diagramShell{
          margin-top: 12px;
          border: 1px dashed rgba(255,255,255,.18);
          background: rgba(255,255,255,.03);
          border-radius: var(--radius2);
          padding: 14px;
          overflow-x: auto;
        }
        .mermaidOut :global(svg){
          max-width: 100%;
          height: auto;
        }
        /* Make Mermaid text readable */
        .mermaidOut svg text{
          fill: rgba(255,255,255,.90) !important;
        }
        .mermaidOut svg .node rect,
        .mermaidOut svg .node polygon,
        .mermaidOut svg .node path{
          stroke: rgba(255,255,255,.18) !important;
        }

        /* Animated architecture */
        .animWrap{
          margin-top: 12px;
          border: 1px dashed rgba(255,255,255,.18);
          background: rgba(255,255,255,.03);
          border-radius: var(--radius2);
          padding: 16px;
          overflow-x: auto;
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .animRow{
          display:flex;
          gap: 10px;
          align-items:center;
          min-width: 1050px;
        }
        .animNode{
          border: 1px solid rgba(255,255,255,.14);
          background: rgba(0,0,0,.22);
          border-radius: 14px;
          padding: 10px 12px;
          min-width: 150px;
          position: relative;
          animation: float 4.5s ease-in-out infinite;
        }
        .animNode b{
          display:block;
          font-size: 13px;
          margin-bottom: 4px;
        }
        .animNode small{
          color: var(--muted);
          font-size: 12px;
        }
        @keyframes float{
          0%,100%{ transform: translateY(0px); }
          50%{ transform: translateY(-3px); }
        }
        .glow1{ box-shadow: 0 16px 40px rgba(139,92,246,.10); }
        .glow2{ box-shadow: 0 16px 40px rgba(34,211,238,.08); }
        .glow3{ box-shadow: 0 16px 40px rgba(52,211,153,.06); }

        .animLink{
          width: 34px;
          height: 2px;
          background: rgba(255,255,255,.22);
          position: relative;
          border-radius: 999px;
          flex: 0 0 34px;
          overflow: hidden;
        }
        .animLink:after{
          content:"";
          position:absolute;
          right:-2px;
          top:50%;
          transform: translateY(-50%);
          width:0; height:0;
          border-left: 7px solid rgba(255,255,255,.28);
          border-top: 5px solid transparent;
          border-bottom: 5px solid transparent;
        }
        .dotRun{
          position:absolute;
          top:50%;
          width: 10px;
          height: 10px;
          border-radius: 50%;
          transform: translate(-12px, -50%);
          background: linear-gradient(135deg, var(--accent), var(--accent2));
          box-shadow: 0 0 0 0 rgba(0, 255, 204,.40);
          animation: run 1.15s linear infinite;
        }
        @keyframes run{
          0%{ left: -10px; box-shadow: 0 0 0 0 rgba(0, 255, 204,.35); }
          70%{ box-shadow: 0 0 0 10px rgba(0, 255, 204,0); }
          100%{ left: 100%; box-shadow: 0 0 0 0 rgba(0, 255, 204,0); }
        }
        .animLegend{
          display:flex;
          flex-wrap: wrap;
          gap: 10px;
          margin-top: 12px;
        }

        .pillGrid{
          display:grid;
          grid-template-columns: repeat(4, minmax(0,1fr));
          gap: 10px;
          margin-top: 10px;
        }
        .pill{
          border: 1px solid rgba(255,255,255,.12);
          background: rgba(255,255,255,.05);
          border-radius: 999px;
          padding: 10px 12px;
          font-size: 13px;
          color: rgba(255,255,255,.80);
          display:flex;
          align-items:center;
          justify-content: space-between;
          gap: 10px;
          min-height: 42px;
        }
        .pill span{
          color: rgba(255,255,255,.6);
          font-size: 12px;
        }

        .btnRow{
          display:flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 12px;
        }
        .btn{
          border: 1px solid rgba(255,255,255,.14);
          background: rgba(255,255,255,.06);
          padding: 10px 12px;
          border-radius: 999px;
          color: rgba(255,255,255,.86);
          font-size: 13px;
          transition: .2s ease;
          display:inline-flex;
          align-items:center;
          gap: 8px;
        }
        .btn:hover{
          background: rgba(255,255,255,.18);
          border-color: rgba(255,255,255,.38);
          color: #fff;
          transform: translateY(-2px) scale(1.03);
          box-shadow: 0 6px 22px rgba(255,255,255,.10), 0 0 0 1px rgba(255,255,255,.10);
          text-decoration: none;
        }
        .btn.primary:hover{
          background: linear-gradient(135deg, rgba(124, 93, 250,.75), rgba(0, 255, 204,.45));
          border-color: rgba(255,255,255,.28);
          box-shadow: 0 8px 28px rgba(124,93,250,.35), 0 0 0 1px rgba(124,93,250,.25);
          transform: translateY(-2px) scale(1.03);
        }
        .btn.primary{
          background: linear-gradient(135deg, rgba(124, 93, 250,.45), rgba(0, 255, 204,.22));
          border-color: rgba(255,255,255,.16);
        }
        .tinyMuted{ margin-top: 12px; font-size: 12.5px; color: rgba(255,255,255,.60); }

        .footer{
          margin-top: 22px;
          padding: 22px;
          border-radius: var(--radius);
          border: 1px solid rgba(255,255,255,.10);
          background: rgba(0,0,0,.15);
          display:flex;
          align-items:flex-start;
          justify-content: space-between;
          gap: 14px;
          flex-wrap: wrap;
        }

        @media (max-width: 920px){
          .heroGrid{ grid-template-columns: 1fr; }
          .pillGrid{ grid-template-columns: repeat(2, minmax(0,1fr)); }
          .grid2{ grid-template-columns: 1fr; }
        }
        @media (max-width: 520px){
          .pillGrid{ grid-template-columns: 1fr; }
          .chip{ font-size: 12px; }
        }
      `}</style>
        </div>
    );
}