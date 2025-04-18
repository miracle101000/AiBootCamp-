# Main backend using FastAPI, vector store, memory, PDF loader
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.memory import ConversationBufferMemory
import os
import glob

load_dotenv()
app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str
    session_id: str

openai_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_key)
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

# Load all docs (PDF and TXT)
def load_all_docs():
    docs = []
    for pdf_path in glob.glob("docs/*.pdf"):
        pdf_loader = PyPDFLoader(pdf_path)
        docs.extend(pdf_loader.load())
    for txt_path in glob.glob("docs/*.txt"):
        text_loader = TextLoader(txt_path)
        docs.extend(text_loader.load())
    return docs

raw_docs = load_all_docs()
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(raw_docs)
vectorstore = FAISS.from_documents(split_docs, embeddings)

# Session memory tracker
memories = {}

def get_memory(session_id: str):
    if session_id not in memories:
        memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    return memories[session_id]

@app.post("/ask")
def ask_question(query: Query):
    memory = get_memory(query.session_id)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    result = qa_chain.run(query.question)
    return {"answer": result}
