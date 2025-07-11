from fastapi import FastAPI
from .api import router as api_router

app = FastAPI(title="FinOps Forecast API")

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Welcome to the FinOps Forecast API!"}
