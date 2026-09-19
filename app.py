import hashlib
import logging
import os
import time
from contextlib import asynccontextmanager

import joblib
import redis
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "300"))
MODEL_DELAY_MS = int(os.getenv("MODEL_DELAY_MS", "0"))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load("model.joblib")
    state["cache"] = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    yield
    state["cache"].close()
    state.clear()


app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    text: str


def cache_key(text: str) -> str:
    return "spam:" + hashlib.sha256(text.encode()).hexdigest()


@app.post("/predict")
def predict(msg: Message, response: Response):
    start = time.perf_counter()
    cache = state["cache"]
    key = cache_key(msg.text)

    status = "MISS"
    label = None
    try:
        label = cache.get(key)
        if label is not None:
            status = "HIT"
    except redis.RedisError as e:
        status = "BYPASS"
        logger.warning("Redis read failed: %s", e)

    if label is None:
        if MODEL_DELAY_MS:
            time.sleep(MODEL_DELAY_MS / 1000)
        label = str(state["model"].predict([msg.text])[0])
        if status == "MISS":
            try:
                cache.set(key, label, ex=CACHE_TTL)
            except redis.RedisError as e:
                logger.warning("Redis write failed: %s", e)

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Cache"] = status
    response.headers["X-Handler-Time-Ms"] = f"{elapsed_ms:.3f}"
    logger.info("cache=%s handler_ms=%.3f", status, elapsed_ms)
    return {"label": label}


@app.get("/healthz")
def healthz():
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok"}
