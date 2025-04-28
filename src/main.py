from fastapi import FastAPI
from src.utils.routers import router


app = FastAPI(
    title="Iris Flower Classification API",
    description="A FastAPI service to classify iris flowers using a PyTorch model trained on the Iris dataset.",
    version="1.0"
)

app.include_router(router)
