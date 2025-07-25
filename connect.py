from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableSequence
import os
from dotenv import load_dotenv


load_dotenv()
GROQ_API_KEY = os.getenv("GROK_API_KEY")


DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"


def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

def build_vector_db(chunks):
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db


def load_llm():
    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.2,
        groq_api_key=GROQ_API_KEY
    )
    return llm


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don’t know the answer, just say that you don’t know. Don’t make up anything.
Only use the given context.

Context: {context}
Question: {question}

Start your answer directly.
"""

QUERY_REPHRASE_PROMPT = PromptTemplate.from_template("""
You are an intelligent assistant. Reformulate the following user question into a clearer, more specific query that can be used for information retrieval.

Original Question: {question}
Improved Question:
""")

CRITIC_PROMPT = PromptTemplate.from_template("""
Evaluate the following answer with respect to the context and question. Mention if the answer is fully grounded in the context or not.

Context: {context}
Question: {question}
Answer: {answer}

Evaluation:
""")

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])


embedding_model = get_embedding_model()
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})


llm = load_llm()
rephrase_chain = QUERY_REPHRASE_PROMPT | llm
critic_chain = CRITIC_PROMPT | llm

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)


def agentic_query_handler(user_query, max_retries=2):
    attempts = 0
    query = user_query
    while attempts < max_retries:
        improved_query = rephrase_chain.invoke({"question": query}).content

        response = qa_chain.invoke({"query": improved_query})
        answer = response["result"]
        context = "\n\n".join([doc.page_content for doc in response["source_documents"]])

        evaluation = critic_chain.invoke({
            "context": context,
            "question": user_query,
            "answer": answer
        }).content

        if "grounded" in evaluation.lower() and "yes" in evaluation.lower():
            return answer
        else:
            query = improved_query
            attempts += 1

    return answer

# Step 8: Run
if __name__ == "__main__":
    documents = load_pdf_files(DATA_PATH)
    text_chunks = create_chunks(documents)
    build_vector_db(text_chunks)

    user_query = input("Your question: ")
    final_answer = agentic_query_handler(user_query=user_query)
    print(final_answer)
