from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from src.database.connection import engine, get_db
from src.database import models

# Create tables in Postgres immediately on startup
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Sentinel-Fin API")

@app.get("/")
def health_check():
    return {
        "status": "active",
        "system": "Sentinel-Fin Fraud Engine",
        "database": "connected"
    }

@app.get("/audit-logs")
def get_logs(db: Session = Depends(get_db)):
    # This proves the DB connection is working
    logs = db.query(models.TransactionAudit).limit(10).all()
    return {"count": len(logs), "data": logs}