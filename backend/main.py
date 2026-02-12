from fastapi import FastAPI
from pydantic import BaseModel
from rag import get_qa_chain

app = FastAPI()
qa_chain = get_qa_chain()

class Question(BaseModel):
    query: str

@app.post("/chat")
def chat(question: Question):
    response = qa_chain.invoke(question.query)
    return {"answer": response["result"]}
