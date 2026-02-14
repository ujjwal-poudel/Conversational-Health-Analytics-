# Frontend API Configuration

## ✅ Updated to EC2 Endpoint

Your frontend is now configured to use the EC2 production endpoint:
**http://18.216.62.242:8000**

## 📝 Files Modified

### 1. **[src/config/api.ts](src/config/api.ts)** (NEW)
- Centralized API configuration
- Single place to switch between local/production

### 2. Updated Components:
- [src/pages/VoiceChat.tsx](src/pages/VoiceChat.tsx)
- [src/pages/TextChat.tsx](src/pages/TextChat.tsx)
- [src/components/Chatbot.tsx](src/components/Chatbot.tsx)
- [src/components/AudioChat.tsx](src/components/AudioChat.tsx)
- [src/components/Questionnaire.tsx](src/components/Questionnaire.tsx)
- [vite.config.ts](vite.config.ts) - Proxy configuration

## 🔄 Switching Between Environments

### To Switch Back to Local Development:

Edit [src/config/api.ts](src/config/api.ts):

```typescript
// Production EC2 endpoint
// export const API_BASE_URL = 'http://18.216.62.242:8000';

// Local development endpoint
export const API_BASE_URL = 'http://localhost:8000';
```

Just comment/uncomment the appropriate line!

### Or Use Environment Variables:

1. Create `.env` file in frontend folder:
```bash
VITE_API_URL=http://18.216.62.242:8000  # EC2
# VITE_API_URL=http://localhost:8000    # Local
```

2. Update [src/config/api.ts](src/config/api.ts):
```typescript
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
```

## 🧪 Testing

### Start Frontend:
```bash
cd frontend
npm run dev
```

### What to Test:
1. **Text Chat** - Navigate to text chat page, start conversation
2. **Audio Chat** - Navigate to audio chat, test voice recording
3. **Questionnaire** - Submit text answers
4. **Health Check** - Visit http://18.216.62.242:8000/health

### Expected Behavior:
- All API calls now go to EC2 instance
- CORS should work (backend allows all origins)
- Models should load successfully on EC2

## 🐛 Troubleshooting

### CORS Errors:
Check browser console. If you see CORS errors, verify EC2 backend CORS settings in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Connection Refused:
- Check if EC2 security group allows inbound on port 8000
- Verify Docker container is running: `docker ps`
- Check backend health: `curl http://18.216.62.242:8000/health`

### Audio Not Playing:
- Check browser console for audio path errors
- Verify audio files are being served from `/audio` endpoint
- EC2 should serve static files from `audio_data/` folder

## 📊 Network Tab

Open browser DevTools → Network tab to see:
- All requests going to `http://18.216.62.242:8000`
- Response times from EC2
- Any 404/500 errors

## 🎯 Ready to Test!

Your frontend is now fully configured to talk to your EC2 backend. Just run:

```bash
npm run dev
```

And visit http://localhost:3001 to test the cloud deployment! 🚀
