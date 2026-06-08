from fastapi import FastAPI

from core.exceptions import register_exception_handlers
from core.logging import configure_logging, register_request_logging

configure_logging("chatbot-api")

from LLM.OSS.Open_AI_OSS import router as Open_AI_OSS
from LLM.OSS.lifecycle import chatbot_lifespan


app = FastAPI(lifespan=chatbot_lifespan)
register_request_logging(app)
register_exception_handlers(app)

@app.get("/health")
def health():
    return {"status":"ok"}

app.include_router(Open_AI_OSS)
