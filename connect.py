from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROK_API_KEY")


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_pdf_files(path):
    loader = DirectoryLoader(path, glob='*.pdf', loader_cls=PyPDFLoader)
    return loader.load()

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# --- Build or Load Vectorstore ---
def build_vector_db(chunks):
    embeddings = get_embedding_model()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)

def load_vector_db():
    embeddings = get_embedding_model()
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)


def load_llm():
    return ChatGroq(model_name="llama3-8b-8192", temperature=0.2, groq_api_key=GROQ_API_KEY)


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, say "This information is not available in the provided context."

Context: {context}
Question: {question}

Start your answer directly:
"""


def create_qa_tool(llm, retriever):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

    def qa_func(inputs):
        response = qa_chain.invoke(inputs)
        answer = response.get("result", "").strip()
        if not answer or "i don't know" in answer.lower():
            return "This information is not available in the provided context."
        return answer

    return Tool(
        name="rag_tool",
        func=qa_func,
        description="Use this tool to answer user questions based on the uploaded PDFs."
    )


def create_rephrase_tool(llm):
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an intelligent assistant. Reformulate the following user question into a clearer, more specific query.

Original Question: {question}
Improved Question:"""
    )

    chain = prompt | llm

    def rephrase_func(inputs):
        return chain.invoke({"question": inputs}).content.strip()

    return Tool(
        name="rephrase_tool",
        func=rephrase_func,
        description="Use this tool to improve ambiguous or unclear questions."
    )


def create_critic_tool(llm):
    prompt = PromptTemplate(
        input_variables=["context", "question", "answer"],
        template="""
Evaluate the following answer with respect to the context and question.
Mention if the answer is fully grounded in the context or not.

Context: {context}
Question: {question}
Answer: {answer}

Evaluation:"""
    )

    chain = prompt | llm

    def critic_func(inputs):
        if not isinstance(inputs, dict) or not all(k in inputs for k in ["context", "question", "answer"]):
            return "Critic tool requires context, question, and answer. The input format was incorrect."
        return chain.invoke(inputs).content.strip()

    return Tool(
        name="critic_tool",
        func=critic_func,
        description="Use this tool to check if the answer is grounded in the retrieved context."
    )


def initialize_agentic_rag(llm, retriever):
    tools = [
        create_qa_tool(llm, retriever),
        create_rephrase_tool(llm),
        create_critic_tool(llm)
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="chat-zero-shot-react-description",
        verbose=True
    )

if __name__ == "__main__":
    if not os.path.exists(DB_FAISS_PATH):
        print("Building vector DB...")
        docs = load_pdf_files(DATA_PATH)
        chunks = create_chunks(docs)
        build_vector_db(chunks)
        print("Vector DB built successfully.")

    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = load_llm()

    agent = initialize_agentic_rag(llm, retriever)

    while True:
        user_input = input("\nAsk a question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        result = agent.invoke({"input": user_input})["output"]
        print("\nAnswer:", result)
