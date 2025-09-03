import streamlit as st
from sqlalchemy import create_engine, Table, Column, Integer, String, ForeignKey, DateTime, Text, MetaData, select, insert, update
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import bcrypt
import os

# LangChain imports
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from dotenv import load_dotenv
load_dotenv()

# ------------------------
# Setup Embeddings & Groq
# ------------------------
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

groq_api_key = os.getenv("API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# ------------------------
# SQLite DB Setup
# ------------------------
engine = create_engine("sqlite:///rag_chat_app.db", connect_args={"check_same_thread": False})
metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow)
)

conversations = Table(
    "conversations", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("title", String, default="New Chat"),
    Column("created_at", DateTime, default=datetime.utcnow)
)

messages = Table(
    "messages", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("conversation_id", Integer, ForeignKey("conversations.id")),
    Column("role", String),
    Column("content", Text),
    Column("timestamp", DateTime, default=datetime.utcnow)
)

metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# ------------------------
# Authentication Helpers
# ------------------------
def signup_user(username, password):
    db = SessionLocal()
    existing = db.execute(select(users).where(users.c.username == username)).fetchone()
    if existing:
        return False, "Username already exists"
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    db.execute(insert(users).values(username=username, password_hash=hashed))
    db.commit()
    return True, "Signup successful"

def login_user(username, password):
    db = SessionLocal()
    user = db.execute(select(users).where(users.c.username == username)).fetchone()
    if user and bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        return True, user.id
    return False, None

# ------------------------
# Session State
# ------------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(page_title="RAG PDF Chat App", layout="wide")
st.title("💬 Conversational RAG with PDF")

# ------------------------
# Login / Signup
# ------------------------
if st.session_state.user_id is None:
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            success, user_id = login_user(username, password)
            if success:
                st.session_state.user_id = user_id
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
    with tab2:
        new_username = st.text_input("New Username", key="signup_user")
        new_password = st.text_input("New Password", type="password", key="signup_pass")
        if st.button("Sign Up"):
            success, msg = signup_user(new_username, new_password)
            if success:
                st.success(msg)
            else:
                st.error(msg)

# ------------------------
# Main App (after login)
# ------------------------
else:
    db = SessionLocal()
    st.sidebar.title("📂 Upload PDF(s)")

    # ------------------------
    # Upload PDF Section (above conversations)
    # ------------------------
    uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            temp_pdf = f"./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

    # ------------------------
    # Sidebar Conversations
    # ------------------------
    st.sidebar.title("Your Conversations")
    convs = db.execute(
        select(conversations).where(conversations.c.user_id == st.session_state.user_id)
    ).fetchall()

    for conv in convs:
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            if st.button(conv.title, key=f"conv_{conv.id}"):
                st.session_state.active_conversation = conv.id
                st.session_state.messages = [
                    {"role": m.role, "content": m.content} 
                    for m in db.execute(select(messages).where(messages.c.conversation_id == conv.id)).fetchall()
                ]
        with col2:
            if st.button("🗑️", key=f"del_{conv.id}"):
                # Delete conversation messages
                db.execute(messages.delete().where(messages.c.conversation_id == conv.id))
                # Delete conversation
                db.execute(conversations.delete().where(conversations.c.id == conv.id))
                db.commit()

                # Clear session if the deleted conversation was active
                if st.session_state.active_conversation == conv.id:
                    st.session_state.active_conversation = None
                    st.session_state.messages = []

                st.success(f"Conversation '{conv.title}' deleted.")
                st.rerun()

    # Create New Conversation (auto-title later)
    if st.sidebar.button("➕ New Chat"):
        result = db.execute(
            insert(conversations).values(user_id=st.session_state.user_id, title="New Chat")
        )
        db.commit()
        st.session_state.active_conversation = result.inserted_primary_key[0]
        st.session_state.messages = []

    if st.sidebar.button("🚪 Logout"):
        st.session_state.user_id = None
        st.session_state.active_conversation = None
        st.session_state.messages = []
        st.rerun()

    # ------------------------
    # Setup RAG Chain
    # ------------------------
    if documents:
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer it, just reformulate if needed."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know, say you don't know. Keep answer concise (max 3 sentences).\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if 'store' not in st.session_state:
                st.session_state.store = {}
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history, input_messages_key="input", history_messages_key="chat_history", output_messages_key="answer"
        )

        # ------------------------
        # Display Chat Messages
        # ------------------------
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # ------------------------
        # Chat Input
        # ------------------------
        if prompt := st.chat_input("Enter your question..."):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Auto-generate conversation title from first user message
            if st.session_state.active_conversation:
                conv_id = st.session_state.active_conversation
                conv_row = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
                if conv_row.title == "New Chat":
                    title_short = prompt if len(prompt) <= 20 else prompt[:30] + "..."
                    db.execute(update(conversations).where(conversations.c.id == conv_id).values(title=title_short))
                    db.commit()

            with st.spinner("Thinking..."):
                session_history = get_session_history(st.session_state.active_conversation)
                response = conversational_rag_chain.invoke({"input": prompt}, config={"configurable": {"session_id": st.session_state.active_conversation}})
                st.chat_message("assistant").write(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

                # Save to SQLite messages table
                if st.session_state.active_conversation:
                    db.execute(
                        insert(messages).values(
                            conversation_id=st.session_state.active_conversation,
                            role="user",
                            content=prompt,
                            timestamp=datetime.utcnow()
                        )
                    )
                    db.execute(
                        insert(messages).values(
                            conversation_id=st.session_state.active_conversation,
                            role="assistant",
                            content=response["answer"],
                            timestamp=datetime.utcnow()
                        )
                    )
                    db.commit() 
