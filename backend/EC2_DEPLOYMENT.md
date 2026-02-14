# EC2 Deployment Guide

## ✅ Issues Fixed

### 1. **Audio Model Paths** 
- Fixed `audio_inference_service.py` to correctly find models in `models/` folder
- Models now load from `backend/models/lasso/` and `backend/models/pca/`

### 2. **Health Endpoint**
- Added `/health` endpoint for load balancer health checks
- Returns status of all loaded models

### 3. **Models Included in Docker Image**
- Updated `.dockerignore` to include models in the build (commented out exclusions)
- Models will be bundled with the Docker image (~390 MB)

## 📦 Deployment Steps

### Step 1: Copy Models to Backend

Make sure these files exist in your backend folder before building:
```bash
backend/models/roberta/model_2_13.pt          # 326 MB
backend/models/lasso/lasso_final_v8/          # 1.1 MB (folder with scaler, selector, model)
backend/models/pca/pca_wav2vec2.joblib        # 1.2 MB
backend/models/piper/en_US-lessac-medium.onnx # 60 MB
```

### Step 2: Build Docker Image on EC2

```bash
cd backend
docker build -t depression-backend:latest .
```

### Step 3: Run Container

```bash
docker run -d \
  --name depression-app \
  -p 8000:8000 \
  --restart unless-stopped \
  depression-backend:latest
```

### Step 4: Verify Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "conversational-health-analytics",
  "models_loaded": {
    "depression_model": true,
    "audio_service": true,
    "conversation_engine": true
  }
}
```

## 🔧 LLM Configuration

### Option 1: Template Fallback (Current - No Setup Needed)
- **Status**: Already working
- **Behavior**: Uses predefined question templates (no LLM paraphrasing)
- **Pros**: No external dependencies, works immediately
- **Cons**: Questions sound more scripted

### Option 2: Install Ollama in Docker
Add to Dockerfile before `USER appuser`:
```dockerfile
RUN curl -fsSL https://ollama.com/install.sh | sh
```

Set environment variable:
```bash
docker run -d \
  -e USE_OLLAMA=true \
  -e OLLAMA_MODEL=llama3.2 \
  ...
```

### Option 3: Use Gemini API (Cloud)
Set environment variable:
```bash
docker run -d \
  -e USE_OLLAMA=false \
  -e GEMINI_API_KEY=your_api_key_here \
  ...
```

## 🌐 Expose to Public (Optional)

### Using nginx reverse proxy:
```bash
sudo apt install nginx
```

Create `/etc/nginx/sites-available/depression-app`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/depression-app /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 📊 Resource Requirements

**Minimum**: t3.medium (2 vCPU, 4 GB RAM)
**Recommended**: t3.large (2 vCPU, 8 GB RAM) for better performance

### Memory Usage:
- Docker Base: ~200 MB
- RoBERTa Model: 326 MB
- Wav2Vec2: 722 MB (auto-downloads on first use)
- faster-whisper: 461 MB (auto-downloads)
- Lasso + PCA: 2.3 MB
- Piper TTS: 60 MB
- **Total**: ~1.77 GB + OS overhead

## 🐛 Troubleshooting

### Check logs:
```bash
docker logs -f depression-app
```

### Models not loading:
```bash
docker exec depression-app ls -lh models/roberta/
docker exec depression-app ls -lh models/lasso/
docker exec depression-app ls -lh models/pca/
```

### Restart container:
```bash
docker restart depression-app
```

### Rebuild from scratch:
```bash
docker stop depression-app
docker rm depression-app
docker rmi depression-backend:latest
docker build -t depression-backend:latest .
docker run -d --name depression-app -p 8000:8000 depression-backend:latest
```
