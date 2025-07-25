
# Agentic RAG: Intelligent PDF-Based Question Answering System

This project is an advanced version of traditional Retrieval-Augmented Generation (RAG), now upgraded to **Agentic RAG**. It is designed to read PDF files (such as lecture slides), understand their content, and intelligently answer user questions.

Instead of just retrieving data based on a fixed query, the system **thinks**, **improves the question**, and **evaluates the answer** — just like a smart assistant.

---

## What This Project Does

- Reads PDFs from the `data/` folder.
- Splits the content into small chunks for better understanding.
- Converts each chunk into vector embeddings using HuggingFace model.
- Stores these vectors in a FAISS vector database.
- When a question is asked:
  1. The system improves the question automatically.
  2. It searches for the most relevant data.
  3. Answers using the LLaMA 3 model via Groq.
  4. Checks if the answer is correct and well-grounded.
  5. If not, it retries with a better version of the question.

---

## Technologies Used

- **LangChain** — for chaining all AI steps
- **Groq + LLaMA 3** — for fast and accurate answers
- **HuggingFace Embeddings** — for converting text into vectors
- **FAISS** — for vector search
- **Python + dotenv** — for scripting and secure key storage

---

## 📁 Project Structure

```
RAG_Project/
├── data/               # Put your PDF slides here
├── vectorstore/        # Vector DB is stored here
├── create.py           # Script to load PDFs and create vector DB
├── connect.py          # Main script that handles the Agentic QA
├── .env                # Store your GROQ_API_KEY here
└── README.md           # Project description
```

---

## ⚙️ How To Run

1. Put your PDF files in the `data/` folder.
2. Add your Groq API key in `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run `create.py` to prepare the database:
   ```bash
   python create.py
   ```
4. Run `connect.py` to ask questions:
   ```bash
   python connect.py
   ```

---

## Example

```
Your question: explain OS
Final Output: An operating system (OS) is software that acts as a bridge between the user and hardware...
```

---

## Learning Reference

I mainly learned Agentic RAG from this article:  
🔗 [https://www.datacamp.com/blog/agentic-rag](https://www.datacamp.com/blog/agentic-rag)  
I also explored a few more blogs and tutorials to understand how agents and feedback loops work with RAG.

---

