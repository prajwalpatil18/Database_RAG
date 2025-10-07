import streamlit as st
from sqlalchemy import create_engine, Table, Column, Integer, String, DateTime, MetaData, select, insert
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import bcrypt
import os

# LangChain imports
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

# PDF handling imports
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PyPDF2 import PdfReader, PdfWriter

# ---------- Database setup ----------
DATABASE_URL = "sqlite:///users.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("username", String, unique=True, nullable=False),
    Column("password", String, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow),
)
metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ---------- PDF Append Helper ----------
def append_text_to_pdf(pdf_path, text_to_append):
    """Appends text to an existing or new PDF."""
    temp_page = "temp_append.pdf"

    # Create a new PDF page with the text
    c = canvas.Canvas(temp_page, pagesize=letter)
    text_object = c.beginText(40, 750)
    text_object.setFont("Helvetica", 11)
    for line in text_to_append.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()

    # Combine the new page with existing pages (if any)
    writer = PdfWriter()
    if os.path.exists(pdf_path):
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                writer.add_page(page)
        except Exception:
            pass

    # Append new content page
    append_reader = PdfReader(temp_page)
    for page in append_reader.pages:
        writer.add_page(page)

    # Save merged PDF
    with open(pdf_path, "wb") as f_out:
        writer.write(f_out)
    os.remove(temp_page)


# ---------- User Authentication ----------
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode("utf-8"), hashed)

def register_user(username, password):
    session = Session()
    if session.query(users).filter_by(username=username).first():
        st.warning("Username already exists. Please choose another.")
        return False
    hashed_pw = hash_password(password)
    new_user = {"username": username, "password": hashed_pw}
    session.execute(insert(users).values(**new_user))
    session.commit()
    st.success("Registration successful! You can now log in.")
    return True

def login_user(username, password):
    session = Session()
    user = session.query(users).filter_by(username=username).first()
    if user and check_password(password, user.password):
        return True
    return False

# ---------- App UI ----------
st.title("🔍 RAG Q&A Streamlit Chat with PDF Logging")

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register":
    st.subheader("Create a New Account")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")
    if st.button("Register"):
        if new_user and new_pass:
            register_user(new_user, new_pass)
        else:
            st.warning("Please fill all fields.")

elif choice == "Login":
    st.subheader("Login to Your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(username, password):
            st.success(f"Welcome, {username}! 🎉")

            # ---------- File Upload ----------
            uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])
            if uploaded_file:
                with open("uploaded.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.info("✅ File uploaded successfully.")

                # Load document for vector retrieval
                loader = PyPDFLoader("uploaded.pdf")
                pages = loader.load()
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(pages, embeddings)
                retriever = vectorstore.as_retriever()

                # Model setup
                llm = ChatOpenAI(model="gpt-3.5-turbo")
                qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

                # Initialize chat
                st.session_state.messages = st.session_state.get("messages", [])

                # Display past messages
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Chat input
                if prompt := st.chat_input("Ask me anything about your document..."):
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    # Append User Prompt to PDF
                    append_text_to_pdf("temp.pdf", f"User: {prompt}\n")

                    # Get Answer
                    answer = qa_chain.run(prompt)
                    st.chat_message("assistant").markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # Append Assistant Response to PDF
                    append_text_to_pdf("temp.pdf", f"Assistant: {answer}\n\n")

                    st.success("📝 Conversation saved to temp.pdf")

        else:
            st.error("Invalid username or password.")
