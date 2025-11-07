from app.database import SessionLocal, engine
from app.models import Question, User, Base
from app.auth import get_password_hash
import uuid

def init_db():
    # Create tables
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # Create admin user
        admin = db.query(User).filter(User.name == "Admin").first()
        if not admin:
            hashed_password = get_password_hash("admin123")
            admin = User(id=str(uuid.uuid4()), name="Admin", hashed_password=hashed_password, is_admin=True)
            db.add(admin)

        # Create sample questions
        questions = [
            "What is your name?",
            "How old are you?",
            "What is your favorite color?",
            "Describe your daily routine.",
            "What are your hobbies?"
        ]

        for i, q_text in enumerate(questions, 1):
            question = db.query(Question).filter(Question.order == i).first()
            if not question:
                question = Question(text=q_text, order=i)
                db.add(question)

        db.commit()
        print("Database initialized with admin user and sample questions.")
    finally:
        db.close()

if __name__ == "__main__":
    init_db()