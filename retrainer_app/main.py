from fastapi import FastAPI
from contextlib import asynccontextmanager

from retrainer_app.core.db import engine, Base
from retrainer_app.retrainer.api.retrainer_api import router as retrainer_router
from retrainer_app.monitor.api.monitor_api import router as monitor_router

def create_tables():
    Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    print("Database tables created (if they didn't exist).")
    yield

app = FastAPI(title="Retrainer Service", lifespan=lifespan)

app.include_router(retrainer_router, prefix="/retrainer", tags=["Retrainer"])
app.include_router(monitor_router, prefix="/monitor", tags=["Monitor"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Retrainer Service"}
