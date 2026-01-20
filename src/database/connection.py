import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Get credentials from Docker Environment
DB_USER = os.environ.get("POSTGRES_USER")
DB_PASS = os.environ.get("POSTGRES_PASSWORD")
DB_NAME = os.environ.get("POSTGRES_DB")
DB_HOST = os.environ.get("POSTGRES_HOST")

if not all([DB_USER, DB_PASS, DB_NAME, DB_HOST]):
    raise EnvironmentError("CRITICAL: .env credentials not found in Container!")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()