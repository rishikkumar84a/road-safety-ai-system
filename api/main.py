from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import prediction
import uvicorn

app = FastAPI(
    title="Road Safety AI System API",
    description="API for detecting vehicles, pedestrians, smoke, and lane drift.",
    version="1.0.0"
)

# CORS Configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
