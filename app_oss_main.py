from fastapi import FastAPI
from LLM.OSS.Open_AI_OSS import router as Open_AI_OSS


app = FastAPI()

@app.get("/health")
def health():
    return {"status":"ok"}

app.include_router(Open_AI_OSS)