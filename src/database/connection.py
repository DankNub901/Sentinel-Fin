import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Check for a single connection string first (Used by GitHub Actions)
DATABASE_URL = os.environ.get("DATABASE_URL")

# 2. If not found, try to build it from individual Docker environment variables
if not DATABASE_URL:
    DB_USER = os.environ.get("POSTGRES_USER")
    DB_PASS = os.environ.get("POSTGRES_PASSWORD")
    DB_NAME = os.environ.get("POSTGRES_DB")
    DB_HOST = os.environ.get("POSTGRES_HOST")
    
    # Only raise the error if BOTH methods fail
    if not all([DB_USER, DB_PASS, DB_NAME, DB_HOST]):
        raise EnvironmentError("CRITICAL: Database credentials not found in Environment!")
        
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

# 3. Create engine and session as normal
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()