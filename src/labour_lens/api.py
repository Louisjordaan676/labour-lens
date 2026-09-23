from fastapi import FastAPI
from pydantic import BaseModel
from .chatbot import ask_labourlens


app = FastAPI(title="LAbourLens API")


@app.get("/")
def root():
    return {"message": "LabourLens API is running"}


class QuestionRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    page: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    return ask_labourlens(question=request.question)
