from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .. import models, database
from pydantic import BaseModel
from typing import List

router = APIRouter()

class QuestionResponse(BaseModel):
    id: int
    text: str
    order: int

@router.get("/questions", response_model=List[QuestionResponse])
def get_questions(db: Session = Depends(database.get_db)):
    questions = db.query(models.Question).order_by(models.Question.order).all()
    return questions