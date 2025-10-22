# Database_RAG

### 📋 Project Overview

This project is a **Streamlit web app** that allows users to chat with an AI assistant and **save all conversations** to a **SQLite database**.
Each chat session is stored separately and can be viewed later from the **sidebar**, where users can switch between different chat sessions seamlessly.

---

### 🚀 Features

✅ **Interactive Chat UI** — Type prompts and get instant AI-generated responses.
✅ **Conversation History** — Each session is stored in the SQLite database.
✅ **Sidebar Navigation** — Easily switch between old chat sessions.
✅ **Automatic Conversation Titles** — The app automatically assigns titles to new conversations.
✅ **Persistent Storage** — All chat data is saved and loaded even after app restarts.
✅ **Lightweight Local Database** — Uses SQLite for local data storage.

---

### 🛠️ Tech Stack

| Component | Technology                                 |
| --------- | ------------------------------------------ |
| Frontend  | [Streamlit](https://streamlit.io)          |
| Backend   | Python                                     |
| Database  | SQLite                                     |
| ORM       | SQLAlchemy                                 |
| AI Model  | OpenAI / Custom LLM Integration (optional) |

---

### 🗂️ Project Structure

```
📁 SQL_Implementation/
│
├── main.py                # Main Streamlit application
├── chat.db                # SQLite database file (auto-created)
├── requirements.txt       # Required dependencies
└── README.md              # Project documentation
```

---

### ⚙️ Installation

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/streamlit-chat-saver.git
cd streamlit-chat-saver
```

#### 2️⃣ Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # For Mac/Linux
venv\Scripts\activate      # For Windows
```

#### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4️⃣ Run the App

```bash
streamlit run main.py
```

---

### 📦 Example `requirements.txt`

```text
streamlit
sqlalchemy
openai
```

*(Include any other dependencies you used.)*

---

### 🧩 How It Works

1. **User starts a new chat** → App creates a new conversation entry in SQLite.
2. **Messages are exchanged** → Each user and AI message is saved in the database.
3. **Sidebar updates automatically** → Shows all chat sessions with timestamps or titles.
4. **User can switch between sessions** → Loads chat history from the database.

---

### 🗃️ Database Schema

**Table: conversations**

| Column     | Type     | Description                       |
| ---------- | -------- | --------------------------------- |
| id         | INTEGER  | Unique conversation ID            |
| title      | TEXT     | Title or name of the conversation |
| created_at | DATETIME | When conversation started         |

**Table: messages**

| Column          | Type     | Description                                |
| --------------- | -------- | ------------------------------------------ |
| id              | INTEGER  | Unique message ID                          |
| conversation_id | INTEGER  | Foreign key referencing `conversations.id` |
| role            | TEXT     | 'user' or 'assistant'                      |
| content         | TEXT     | Actual message content                     |
| timestamp       | DATETIME | When message was sent                      |

---

### 🧠 Future Improvements

🔹 Add user authentication (login system)
🔹 Add AI model selection dropdown (e.g., GPT-4, Llama, Gemini)
🔹 Allow exporting conversation as PDF or text file
🔹 Improve UI with message bubbles and colors

---

### 👨‍💻 Author

**Prajwal Patil**
💼 IT Professional | ⚙️ AI & Data Engineering Enthusiast
📧 Email: *[your email here]*

---

### 📄 License

This project is licensed under the **MIT License** – feel free to use and modify it.

---

Would you like me to:

* ✅ make a **ready-to-download `README.md` file**, or
* 📘 add **screenshots and usage examples** (for GitHub visuals)?
