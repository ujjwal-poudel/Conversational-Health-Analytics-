from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import engine
from app import models
from app.routers import auth_router, question_router, answer_router, admin_router

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Audio Question App", description="API for audio question submissions with transcription")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router, prefix="/auth", tags=["auth"])
app.include_router(question_router.router, tags=["questions"])
app.include_router(answer_router.router, tags=["answers"])
app.include_router(admin_router.router, prefix="/admin", tags=["admin"])

@app.get("/")
def read_root():
    return {"message": "Audio Question App API"}