from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.dependencies import warmup_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs warmup_models() once at startup before the first request.
    Models are loaded into the singleton cache in inference/model.py.
    Nothing to teardown — YOLO models live in process memory.
    """
    warmup_models()
    yield


app = FastAPI(
    title       = "SARFuse API",
    description = "Dual-stream SAR + Optical fusion inference server",
    version     = "0.3.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

app.include_router(router, prefix="/api/v1")