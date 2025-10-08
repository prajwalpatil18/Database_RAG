import streamlit as st
from sqlalchemy import create_engine, Table, Column, Integer, String, ForeignKey, DateTime, Text, MetaData, select, insert, update, delete
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import bcrypt
import os


# Avoid gRPC and Chroma lock noise
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

# LangChain imports
# --- REPLACED Chroma with FAISS to avoid file locking ---
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
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
if "language" not in st.session_state:
    st.session_state.language = "English"
if "store" not in st.session_state:
    st.session_state.store = {}

st.set_page_config(page_title="RAG PDF Chat App", layout="wide")
st.title("💬 Conversational RAG with PDF and Language Selection")



from PyPDF2 import PdfReader, PdfWriter # type: ignore
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import tempfile


def append_text_to_pdf(pdf_path, new_text):
    # Create a temporary PDF with the new text
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(temp_pdf.name, pagesize=letter)
    c.drawString(100, 750, new_text)
    c.save()

    # Read the original and new PDFs
    reader = PdfReader(pdf_path)
    new_reader = PdfReader(temp_pdf.name)
    writer = PdfWriter()

    # Copy all old pages
    for page in reader.pages:
        writer.add_page(page)

    # Add the new page with text
    writer.add_page(new_reader.pages[0])

    # Overwrite the original PDF
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Clean up temp file
    os.remove(temp_pdf.name)




from fpdf import FPDF
import os

os.makedirs("data", exist_ok=True)
pdf_path = "data.pdf"

pdf = FPDF()
pdf.add_page()               # Add one empty page
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, txt="")      # Optional placeholder text
pdf.output(pdf_path)

print(f"Created valid PDF at {pdf_path}")


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
    #st.sidebar.title("📄 Upload PDF(s)")

    # ------------------------
    # Language Selection
    # ------------------------
    st.sidebar.subheader("🌐 Choose Response Language")
    st.session_state.language = st.sidebar.radio(
        "Select a language for responses:",
        ["English", "Hindi"],
        index=0 if st.session_state.language == "English" else 1
    )

    # ------------------------
    # Upload PDFs
    # ------------------------
    #uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    #documents = []
    path = f"./data.txt"
    loader = TextLoader(path)
    docs = loader.load()
    #documents.extend(docs)
    print(f"Documents loaded: {len(docs)}")
    print("Document content preview:")
    print(docs[0].page_content[:500])  # show first 500 chars


    

    # ------------------------
    # Build Vectorstore using FAISS (no locking)
    # ------------------------
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(f"Splits created: {len(splits)}")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # ------------------------
    # Sidebar Conversations
    # ------------------------
    st.sidebar.title("💬 Your Conversations")
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
                db.execute(delete(messages).where(messages.c.conversation_id == conv.id))
                db.execute(delete(conversations).where(conversations.c.id == conv.id))
                db.commit()
                if st.session_state.active_conversation == conv.id:
                    st.session_state.active_conversation = None
                    st.session_state.messages = []
                st.success(f"Conversation '{conv.title}' deleted.")
                st.rerun()

    # Create New Conversation
    if st.sidebar.button("➕ New Chat"):
        result = db.execute(insert(conversations).values(user_id=st.session_state.user_id, title="New Chat"))
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
    if docs:
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, "
            "formulate a standalone question that can be understood "
            "without chat history. Do NOT answer, just reformulate if needed."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        # ✅ System prompt includes selected language
        system_prompt = (
            f"You are an assistant for question-answering tasks. "
            f"Use the text file content to answer the question only. "
            f"If the answer is not in the text file say don't know strictly in {st.session_state.language} language.. Keep the answer concise (max 3 sentences). "
            f"Always respond in {st.session_state.language} language.\n\n{{context}}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # ------------------------
        # Display Chat
        # ------------------------
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # ------------------------
        # Chat Input
        # ------------------------
        lang_placeholder = (
            "Enter your question..." if st.session_state.language == "English" else "अपना प्रश्न दर्ज करें..."
        )
        if prompt := st.chat_input(lang_placeholder):
            st.chat_message("user").write(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            #append_text_to_pdf("./data.pdf", prompt)

            # Update conversation title if new
            if st.session_state.active_conversation:
                conv_id = st.session_state.active_conversation
                conv_row = db.execute(select(conversations).where(conversations.c.id == conv_id)).fetchone()
                if conv_row.title == "New Chat":
                    title_short = prompt if len(prompt) <= 30 else prompt[:30] + "..."
                    db.execute(update(conversations).where(conversations.c.id == conv_id).values(title=title_short))
                    db.commit()

            with st.spinner("Thinking..."):
                session_history = get_session_history(str(st.session_state.active_conversation))
                response = conversational_rag_chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": str(st.session_state.active_conversation)}}
                )
                answer = response["answer"]
                st.chat_message("assistant").write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Save chat to DB
                if st.session_state.active_conversation:
                    db.execute(insert(messages).values(
                        conversation_id=st.session_state.active_conversation,
                        role="user",
                        content=prompt,
                        timestamp=datetime.utcnow()
                    ))
                    db.execute(insert(messages).values(
                        conversation_id=st.session_state.active_conversation,
                        role="assistant",
                        content=answer,
                        timestamp=datetime.utcnow()
                    ))
                    db.commit()
