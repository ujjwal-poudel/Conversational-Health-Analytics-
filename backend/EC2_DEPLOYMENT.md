# EC2 Deployment Guide

This guide details the steps to deploy the backend to an AWS EC2 instance. The process involves completely cleaning the existing Docker environment, uploading fresh code, and rebuilding the container.

## Prerequisites

- **Groq API Key:** `<YOUR_GROQ_API_KEY>`
- **SSH Key:** `ml-key.pem` (must be present in your local terminal directory)
- **EC2 IP Address:** `<YOUR_EC2_IP>`
- **User:** `ubuntu`

---

## 0. Connect to EC2

From your local terminal (ensure `ml-key.pem` is in the current directory):

```bash
ssh -i ml-key.pem ubuntu@<YOUR_EC2_IP>
```

---

## 1. Clean Docker Environment

On the EC2 instance, remove all existing containers, images, volumes, and build cache to ensure a clean slate.

```bash
# Stop all running containers
docker ps -aq | xargs -r docker stop

# Remove all containers
docker ps -aq | xargs -r docker rm -f

# Remove all images
docker images -aq | xargs -r docker rmi -f

# Remove all volumes
docker volume ls -q | xargs -r docker volume rm -f

# Prune system (networks, images, build cache)
docker system prune -af --volumes
docker builder prune -af
```

Verify everything is clean:

```bash
docker ps -a
docker images
docker system df
df -h
```

---

## 2. Remove Project Files

On the EC2 instance, remove the previous project directories.

```bash
rm -rf ~/backend ~/app
```

Optional: If you need to clear persisted data (models/user_data), run:

```bash
sudo rm -rf /opt/models /opt/user_data
```

---

## 3. Upload Fresh Code

Exit the EC2 session to return to your local machine:

```bash
exit
```

From your local terminal, upload the backend code to the EC2 instance:

```bash
scp -i ml-key.pem -r \
/path/to/your/local/backend \
ubuntu@<YOUR_EC2_IP>:~
```

This will create a `~/backend` directory on the EC2 instance containing your latest code.

---

## 4. Build and Run

SSH back into the EC2 instance:

```bash
ssh -i ml-key.pem ubuntu@<YOUR_EC2_IP>
```

### Setup Persistent Directories

Create and set permissions for directories that will persist data outside the container:

```bash
sudo mkdir -p /opt/models /opt/user_data
sudo chown -R 1000:1000 /opt/models /opt/user_data
sudo chmod -R 775 /opt/models /opt/user_data
```

### Configure Environment

Navigate to the backend directory and set up the environment variables:

```bash
cd ~/backend
cp .env.example .env
```

Open `.env` to add your API keys and configuration:

```bash
nano .env
```

**Required changes in `.env`:**
- Set `GROQ_API_KEY` to your key
- Ensure `FIREBASE_CREDENTIALS_PATH` is correct (filename matches upload)

### Build Docker Image

```bash
docker build -t depression-app:latest .
```

### Run Container

Start the container with volume mounts for persistence:

```bash
docker run -d --name depression-app \
  -p 8000:8000 \
  -v /opt/user_data:/app/user_data \
  -v /opt/models:/opt/models \
  depression-app:latest
```

### Verify Deployment

Check that the container is running and view logs:

```bash
docker ps
docker logs -f depression-app
```

The application will be available at: `http://<YOUR_EC2_IP>:8000`
Docs available at: `http://<YOUR_EC2_IP>:8000/docs`
