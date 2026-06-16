from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from core.exceptions import register_exception_handlers
from core.logging import configure_logging, register_request_logging

configure_logging("chatbot-api")

from LLM.OSS.Open_AI_OSS import router as Open_AI_OSS
from LLM.OSS.lifecycle import chatbot_lifespan, get_chatbot_runtime_readiness


app = FastAPI(lifespan=chatbot_lifespan)
register_request_logging(app)
register_exception_handlers(app)
READY_RESPONSES = {
    200: {"description": "Chatbot runtime dependencies are ready"},
    503: {"description": "One or more chatbot runtime dependencies are not ready"},
}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready", responses=READY_RESPONSES)
async def ready():
    payload = get_chatbot_runtime_readiness()
    status_code = status.HTTP_200_OK if payload["status"] == "ready" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=status_code, content=payload)


app.include_router(Open_AI_OSS)
