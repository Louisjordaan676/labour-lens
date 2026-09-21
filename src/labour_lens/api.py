from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="LAbourLens API")


@app.get("/")
def root():
    return {"message": "LabourLens API is running"}


class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    return request.question
