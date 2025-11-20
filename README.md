# Conversational Health Analytics

This is the official repository for the AI Capstone Project, "Conversational Health Analytics." This project aims to develop a full-stack application that leverages multi-modal AI to analyze conversational data (text and audio) to identify linguistic and acoustic biomarkers associated with depression, serving as an assistive tool for clinicians.

## Tech Stack
- **Backend**: FastAPI, Python
- **Frontend**: React, Vite, TypeScript
- **AI/ML**: PyTorch, OpenAI Whisper, Transformers
- **Database/Infrastructure**: SQLAlchemy, Docker

---

## Getting Started

Follow these steps to get the project running on your local machine.

### Prerequisites
Ensure you have the following software installed:
* [Git](https://git-scm.com/downloads)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/ujjwal-poudel/Conversational-Health-Analytics-.git
cd Conversational-Health-Analytics-
```

### 2. Local Development Setup

**Backend Setup:**
1. Navigate to the project root.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
3. Activate the virtual environment:
   - macOS/Linux: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`
4. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

**Frontend Setup:**
1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install dependencies:
   ```bash
   npm install
   # OR if you use yarn
   yarn
   ```
3. Start the development server:
   ```bash
   npm run dev
   ```

---

## Contribution Workflow

We follow a feature branching workflow. Please follow these steps for every contribution.

### 1. Create a Branch
**Naming Convention:**
- Features: `feature/<your-name>/<short-description>`
- Fixes: `fix/<your-name>/<issue-description>`

**Create and switch to your branch:**
```bash
# Make sure you are on main and up to date
git checkout main
git pull origin main

# Create your branch
git checkout -b feature/ujjwal/update-readme
```

### 2. Make Changes and Commit
Make your code changes. When ready, stage and commit them.

```bash
# Stage all changes
git add .

# OR stage specific files
git add README.md

# Commit with a descriptive message
git commit -m "docs: Update README with contribution guidelines"
```

### 3. Push to GitHub
Push your branch to the remote repository.

```bash
# The first time you push a new branch
git push --set-upstream origin feature/ujjwal/update-readme

# Subsequent pushes
git push
```

### 4. Create a Pull Request (PR)
1. Go to the [repository on GitHub](https://github.com/ujjwal-poudel/Conversational-Health-Analytics-).
2. You should see a prompt to "Compare & pull request" for your recently pushed branch.
3. Click it, fill in the details of your changes, and assign a reviewer.
4. Wait for approval before merging into `main`.

---

## Project Structure

* **`backend/`**: FastAPI application and AI model logic.
* **`frontend/`**: React + Vite user interface.
* **`data/`**: Dataset storage (git-ignored).
* **`notebooks/`**: Jupyter notebooks for EDA and experiments.
* **`docs/`**: Project documentation.
* **`Dockerfile`**: Environment definition.