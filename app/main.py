from fastapi import FastAPI
from pydantic import BaseModel
from .rag import retrieve, generate, Answer


app = FastAPI(title="mini-rag")


class Ask(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask", response_model=Answer)
def ask(body: Ask):
    ctx = retrieve(body.question)
    return generate(body.question, ctx)