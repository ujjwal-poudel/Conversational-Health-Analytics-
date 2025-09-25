# Conversational Health Analytics

This is the official repository for the AI Capstone Project, "Conversational Health Analytics." This project aims to develop a full-stack application that leverages multi-modal AI to analyze conversational data (text and audio) to identify linguistic and acoustic biomarkers associated with depression, serving as an assistive tool for clinicians. [cite_start]This project is developed to meet the requirements of the COMP 385 AI Capstone Project[cite: 1].

---

## Getting Started (One-Time Setup)

Follow these steps to get the project running on your local machine for the first time. [cite_start]This ensures a consistent and reproducible setup for all team members, a key requirement for this project[cite: 201].

### Step 1: Prerequisites

Ensure you have the following software installed on your computer:
* [Git](https://git-scm.com/downloads)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Step 2: Clone the Repository

First, clone this repository to your computer using the repository's URL:
```bash
git clone https://github.com/ujjwal-poudel/Conversational-Health-Analytics-.git
```

### Step 3: Navigate into the Directory

```bash
cd Conversational-Health-Analytics
```

### Step 4: Create and Activate Virtual Environment
A virtual environment ensures that all Python packages for this project are kept separate from your global Python installation.
```bash
python3 -m venv venv

# On mac
source venv/bin/activate

# On windows/linux
venv\Scripts\activate

# Install the requirement.txt (Just for your local machine)
pip install -r backend/requirements.txt
```

### Step 4: Build the Docker Image

Make sure Docker Desktop is running. Then, build the Docker image using the following command. This may take a few minutes the first time as it downloads the base images and installs all dependencies.
```bash
docker build -t conversational-health-analytics .
```

---

## Daily Development Workflow

Follow these steps each time you start working on the project.

### Step 1: Get the Latest Updates

Before you start working, always pull the latest changes from the `main` branch to make sure your local repository is up to date with your teammates' work.
```bash
# First, ensure you are on the main branch
git checkout main

# Then, pull the latest changes
git pull origin main

# Create new branch here
# Example branch name: feature/your-name/data-preprocessing
git checkout -b feature/your-name/task-description
```

### Step 2: Check for Environment Changes & Build if Necessary
This is an important check. Look to see if the git pull in Step 1 updated the requirements.txt file.
If requirements.txt was NOT changed, you can skip this step.
If requirements.txt WAS changed, you must rebuild your Docker image to install the new dependencies:
```bash
docker build -t conversational-health-analytics .
```

### Step 3: Run the Project Environment

This command will start the Docker container, which runs the JupyterLab server and syncs your local project folder.

**On macOS or Linux:**
```bash
docker run -p 8888:8888 -v "${PWD}":/app conversational-health-analytics
```

**On Windows (Command Prompt):**
```bash
docker run -p 8888:8888 -v "%CD%":/app conversational-health-analytics
```

### Step 4: Access Jupyter for EDA

[cite_start]You can now access the Jupyter environment for your **Exploratory Data Analysis (EDA)**[cite: 228].

**Option A: Using Your Web Browser**
1.  Look for the URL in your terminal that looks like this: `http://127.0.0.1:8888/lab?token=a1b2c3d4e5f6...`
2.  Copy the **entire URL** and paste it into your web browser.
3.  Navigate to the `notebooks/` folder to start your analysis.

**Option B: Using VS Code (Recommended)**
1.  **Prerequisites:** Ensure you have these extensions > [Docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) and [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) extensions from Microsoft installed in VS Code.
2.  **Copy the URL** (As said in option A) with the token from your terminal.
3.  In VS Code, open your `.ipynb` notebook, click **"Select Kernel"** > **"Existing Jupyter Server,"** and paste the URL (Remember this is not the url of the extensions).

### Step 5: Stop the Container

When you are finished with your work session, you need to stop the running container.

1.  Go to the terminal window where the container is running.
2.  Press **`Ctrl + C`** on your keyboard.

This will gracefully shut down the Jupyter server and stop the container.

---

## Git & Contribution Workflow

To ensure we can all work in parallel without conflicts, please follow this feature branching workflow. [cite_start]This is a standard team collaboration method for projects using GitHub[cite: 104].

### Step 1: Make and Save Changes (Add & Commit)

Work on your task. When you are ready to save your progress, you need to "add" your changed files and "commit" them with a clear message.

```bash
# First update the requirement.txt file, if you add any new packages
# Also let the team know, you added new package and tell them to run docker build when they start
pip freeze > backend/requirements.txt

# To add ALL changed files in the current directory
git add .

# OR to add a single specific file
git add path/to/your/file.py

# Commit your changes with a clear, descriptive message
git commit -m "feat: Add initial data cleaning script for transcripts"
```

### Step 2: Push Your Branch to GitHub

The first time you push a new branch, you need to set its "upstream" remote.
```bash
git push --set-upstream origin feature/your-name/task-description
```
For any subsequent pushes on the same branch, you can simply use `git push`.

### Step 3: Open a Pull Request

Go to the GitHub repository in your browser. GitHub will automatically detect your new branch and prompt you to **"Open a Pull Request."** Fill out the details, assign at least one other team member to review your code, and then merge it into `main` after approval.

---

## Project Structure

[cite_start]This project follows a standard full-stack application structure, as required for the final submission[cite: 266].

* **`backend/`**: Contains all Python code for the AI model and the API.
* **`frontend/`**: Contains all code for the user interface.
* **`data/`**: Used for storing the dataset files (this folder is ignored by Git).
* **`notebooks/`**: Contains Jupyter notebooks for exploration and analysis.
* **`docs/`**: Contains all project documentation, such as the proposal and reports.
* **`Dockerfile`**: The recipe for building our consistent Docker environment.