from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from .. import models, database, auth
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class UserSummary(BaseModel):
    uuid: str
    name: str
    answers_count: int

class AnswerDetail(BaseModel):
    id: int
    question_text: str
    audio_file_path: str
    transcript_text: Optional[str]
    created_at: str

class UserAnswers(BaseModel):
    user: UserSummary
    answers: List[AnswerDetail]

@router.get("/users", response_model=List[UserSummary])
def list_users(db: Session = Depends(database.get_db), current_admin: models.User = Depends(auth.get_current_admin)):
    users = db.query(models.User).filter(models.User.is_admin == False).all()
    result = []
    for user in users:
        answers_count = db.query(models.Answer).filter(models.Answer.user_uuid == user.id).count()
        result.append({
            "uuid": user.id,
            "name": user.name,
            "answers_count": answers_count
        })
    return result

@router.get("/answers", response_model=List[UserAnswers])
def get_all_answers(
    user_uuid: Optional[str] = Query(None),
    question_id: Optional[int] = Query(None),
    db: Session = Depends(database.get_db),
    current_admin: models.User = Depends(auth.get_current_admin)
):
    query = db.query(models.Answer).join(models.User).join(models.Question)

    if user_uuid:
        query = query.filter(models.Answer.user_uuid == user_uuid)
    if question_id:
        query = query.filter(models.Answer.question_id == question_id)

    answers = query.all()

    # Group by user
    user_answers = {}
    for answer in answers:
        user_uuid = answer.user_uuid
        if user_uuid not in user_answers:
            user_answers[user_uuid] = {
                "user": {
                    "uuid": answer.user.id,
                    "name": answer.user.name,
                    "answers_count": 0  # Will be calculated
                },
                "answers": []
            }
        user_answers[user_uuid]["answers"].append({
            "id": answer.id,
            "question_text": answer.question.text,
            "audio_file_path": answer.audio_file_path,
            "transcript_text": answer.transcript_text,
            "created_at": answer.created_at.isoformat()
        })

    # Calculate answers count
    for user_uuid, data in user_answers.items():
        data["user"]["answers_count"] = len(data["answers"])

    return list(user_answers.values())