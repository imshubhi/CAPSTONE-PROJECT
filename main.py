import warnings
warnings.filterwarnings("ignore")


from langchain_community.document_loaders import PyPDFLoader
import os
pdf_folder = "data"
all_documents=[]
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=250,
    length_function=len,
    is_separator_regex=False,
)
text_split=text_splitter.split_documents(all_documents)


from langchain_huggingface.embeddings import HuggingFaceEmbeddings
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# Extract page_content from Document objects
page_content = [doc.page_content for doc in text_split]
model_kwargs = {'device': 'cpu'}
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",model_kwargs=model_kwargs)
embed_results=embeddings.embed_documents(page_content)


from pinecone import Pinecone
from pinecone import ServerlessSpec

pc = Pinecone(pinecone_api_key=os.getenv("PINECONE_API_KEY"))
index_name = "adv-rag-vscode-hybrid-search"

# create the index
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # dimensionality of dense model
        metric="dotproduct",  # sparse values supported only for dotproduct
        spec=ServerlessSpec(
            cloud="aws", 
            region="us-east-1"
        ),
        deletion_protection="disabled",
        tags={
              "environment":"development"
        }
    )
index=pc.Index(index_name)

from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder

bm25_encoder=BM25Encoder().default()
bm25_encoder.fit(page_content)
bm25_encoder.dump("bm25_values.json")
bm25_encoder=BM25Encoder().load("bm25_values.json")
retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)
retriever.add_texts(page_content)


from langchain_groq import ChatGroq

os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

expand_prompt = PromptTemplate.from_template("""
You are an expert assistant that helps users clarify and expand their search queries by rephrasing them with synonyms, 
related concepts, and added context while keeping the meaning intact.

Only return the expanded query as a single sentence, with no explanation or formatting.

Original Query: "{user_query}"
Expanded Query:
""")

expand_chain: Runnable = expand_prompt | llm


decompose_prompt = PromptTemplate.from_template("""
You are an AI assistant. Given an expanded query, break it down into simpler sub-queries only if needed. 
Return ONLY a raw Python list of string sub-queries without any explanations or formatting.

- If the query is simple, still return it as a list with one element.
- Do NOT add any text like "Here are the sub-queries".
- Output must be a valid Python list of strings.

Expanded Query: "{expanded_query}"
""")

parser = StrOutputParser()
decompose_chain: Runnable = decompose_prompt | llm |parser


import json
import ast

def expand_and_decompose_query(query: str) -> list:
    expanded = expand_chain.invoke({"user_query": query})
    print(type(expanded))
    print("Expanded :",expanded)
    decomposed = decompose_chain.invoke({"expanded_query": expanded})
    print(type(decomposed))
    print("Decomposed :", decomposed)

    print("Trying to convert to python list:")
    try: 
        sub_queries = ast.literal_eval(decomposed)
        print(type(sub_queries))
        return sub_queries
    except (SyntaxError, ValueError) as e:
        print("Error parsing decomposed output:", e)


from langchain_core.documents import Document

def retrieve_documents(sub_queries: list, retriever) -> list[Document]:

    all_docs = []
    for sub_q in sub_queries:
        docs = retriever.invoke(sub_q)
        all_docs.extend(docs)
    print(f"Retrieved {len(all_docs)} documents before deduplication.")
    return all_docs


def unique_docs(all_docs:list)-> list[Document]:
    unique_docs_dict = {}
    for doc in all_docs:
        key = doc.page_content.strip()
        score = doc.metadata.get("score", 0.0)
        # Keep the higher score for duplicate page content
        if key not in unique_docs_dict or score > unique_docs_dict[key].metadata.get("score", 0.0):
            unique_docs_dict[key] = doc
    unique_docs = list(unique_docs_dict.values())
    print(f"{len(unique_docs)} unique documents after deduplication.")
    return unique_docs

def docs_rerank(unique_docs: list):
    ranked_docs = sorted(unique_docs, key=lambda d: d.metadata.get("score", 0.0), reverse=True)
    return ranked_docs

def format_context_for_llm(ranked_docs:list[Document]):
    ranked_docs=ranked_docs
    context="\n\n" .join([doc.page_content for doc in ranked_docs])
    return context


final_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Based on the following information from internal documents, answer the user's question clearly and accurately.

Context:
{context}

Question:
{question}

Answer:
""")

final_chain: Runnable = final_prompt | llm

def get_answer_from_ranked_docs(user_query:str,context:str):
    response = final_chain.invoke({
        "context": context,
        "question": user_query
    })
    if hasattr(response, "content"):
        return response.content.strip()
    else:
        return str(response).strip()


def rag_pipeline(user_query: str) -> str:
    sub_queries = expand_and_decompose_query(user_query)
    retrieved_docs = retrieve_documents(sub_queries, retriever)
    deduped_docs = unique_docs(retrieved_docs)
    ranked_docs = docs_rerank(deduped_docs)
    context = format_context_for_llm(ranked_docs)
    answer = get_answer_from_ranked_docs(user_query, context)
    return answer

import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Text, ForeignKey, DateTime,ForeignKey, TIMESTAMP, UUID, Integer
from sqlalchemy.sql import func
from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy_utils import database_exists, create_database
import uuid
import os

load_dotenv()

USER = os.getenv("USER")
PASSWORD = os.getenv("PASSWORD")
HOST = os.getenv("HOST")
PORT = os.getenv("PORT")
DB_NAME = os.getenv("DB_NAME")

TARGET_DATABASE_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}"

if not database_exists(TARGET_DATABASE_URL):
    print(f"Database '{DB_NAME}' does not exist. Creating now...")
    create_database(TARGET_DATABASE_URL)
    print("Database created successfully.")
else:
    print(f"Database '{DB_NAME}' already exists.")

engine = create_engine(TARGET_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    chat_history = relationship("ChatHistory", back_populates="user",cascade="all, delete")

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.user_id'))
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    user = relationship("User", back_populates="chat_history")

Base.metadata.create_all(bind=engine)

# Return User if user is already in database or create new user
def get_or_create_user(username: str, password: str):
    session = SessionLocal()
    user = session.query(User).filter_by(username=username).first()
    if user is None:
        user = User(username=username, password_hash=password)
        session.add(user)
        session.commit()
        session.refresh(user)
    session.close()
    return user


# Store chat history 
def store_chat_history(user_id, query, response):
    session = SessionLocal()
    if hasattr(response, "content"):
        response_text = response.content
    else:
        response_text = str(response)
    chat = ChatHistory(user_id=user_id, query=query, response=response_text)
    session.add(chat)
    session.commit()
    session.refresh(chat)
    session.close()

# Retrieve user history
def get_user_history(user_id, limit=5):
    session = SessionLocal()
    history = (
        session.query(ChatHistory)
        .filter_by(user_id=user_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
        .all()
    )
    session.close()
    return history


get_or_create_user("pulkit","pulkit")

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema import AIMessage

reformulate_prompt = PromptTemplate.from_template("""
You are a helpful assistant. The user has asked previous questions with responses as shown. 
Using this history, reformulate the new query if necessary for more context-aware answers.
Provide only the reformulated query in response and nothing else.
                                                  
Chat History:
{history}

New Query:
{query}

Reformulated Query:
""")

reformulate_chain: Runnable = reformulate_prompt | llm

def reformulate_query(user_query, history_docs):
    history_text = "\n".join([f"Q: {h.query}\nA: {h.response}" for h in history_docs])
    reformulated = reformulate_chain.invoke({"history": history_text, "query": user_query})
    print("Reformulated query type: ",type(reformulated))
    print("Reformulated query: ",reformulated)
    if isinstance(reformulated, AIMessage):
        return reformulated.content.strip()
    else:
        return reformulated.strip()


def rag_pipeline_with_history(username: str, password: str, user_query: str) -> str:
    user = get_or_create_user(username, password)
    
    history = get_user_history(user.user_id)
    reformulated_query = reformulate_query(user_query, history)

    sub_queries = expand_and_decompose_query(reformulated_query)
    retrieved_docs = retrieve_documents(sub_queries, retriever)
    deduped_docs = unique_docs(retrieved_docs)
    ranked_docs = docs_rerank(deduped_docs)
    context = format_context_for_llm(ranked_docs)
    answer = get_answer_from_ranked_docs(user_query, context)

    # Store the original query and response
    store_chat_history(user.user_id, user_query, answer)

    return answer


def chatbot():
    print("Welcome to the RAG-powered chatbot!")
    print("Type your questions below. Type 'exit' or 'quit' to end the session.\n")

    # Get user credentials once
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    while True:
        user_query = input("\nYour question: ")
        if user_query.lower().strip() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            response = rag_pipeline_with_history(username, password, user_query)
            print("\nAnswer:\n", response)
        except Exception as e:
            print("An error occurred:", e)

# Run the chatbot
if __name__ == "__main__":
    chatbot()