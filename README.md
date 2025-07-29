#  Agentic RAG: Smart PDF-Based Question Answering System with LangChain Agents

This project is an enhanced version of traditional **Retrieval-Augmented Generation (RAG)**, upgraded to **Agentic RAG** using **LangChain Agents**.  
It can read PDF files (e.g. lecture slides), understand them, and intelligently answer user questions using multiple tools — just like a thinking assistant.

---

##  What’s New in Agentic RAG?

Unlike traditional RAG systems that simply retrieve and answer, **Agentic RAG adds reasoning, flexibility, and feedback**.  
An **LLM-powered agent** decides what tool to use based on your input.

###  Tools used by the Agent:

1. **RAG Tool**  
   Retrieves relevant chunks from the PDFs and answers questions.

2. **Rephrase Tool**  
   Automatically improves vague or unclear questions before searching.

3. **Critic Tool**  
   Checks if the answer is based on the actual document or not.

All tools are controlled by a single **LangChain Agent** using the `chat-zero-shot-react-description` strategy.

---

##  Technologies Used

- **LangChain Agents** – for decision-making and multi-step execution  
- **Groq + llama3-8b-8192** – for ultra-fast and accurate LLM responses  
- **HuggingFace Embeddings** – for converting text into vector format  
- **FAISS** – for efficient vector-based document search  
- **Python + dotenv** – for logic and key management

---

##  Folder Structure

```
Agentic_RAG_Project/
├── data/               # Add your PDFs here
├── vectorstore/        # Vector DB created here
├── create.py           # Runs once to create vector DB
├── connect.py          # Main Agentic RAG script
├── .env                # Store API keys (not pushed to GitHub)
└── README.md           # Project overview
```

---

##  How to Run the Project

1. Place your PDF documents in the `data/` folder.

2. Add your API key to a `.env` file:
   ```
   GROQ_API_KEY=your_groq_key_here
   ```

3. Create the vector database:
   ```bash
   python create.py
   ```

4. Start the agent:
   ```bash
   python connect.py
   ```

Then just type your question and watch the agent decide what steps to take!

---

##  Example Interaction

```
You: what is os
Agent uses Rephrase Tool → Improved Question: "Explain what an operating system is."
Agent uses RAG Tool → Finds answer from PDF
Agent uses Critic Tool → Confirms answer is correct

Final Answer: An operating system (OS) is software that acts as a bridge between the user and hardware...
```

---
