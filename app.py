import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
import time

# Define document storage folder
documents_folder = "documents_tmp"

# Ensure the directory exists and clear only its contents if it already exists
if os.path.exists(documents_folder):
    for file in os.listdir(documents_folder):
        file_path = os.path.join(documents_folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except PermissionError:
            st.error(f"Permission denied: Unable to delete {file_path}. Check file access permissions.")
os.makedirs(documents_folder, exist_ok=True)

st.title("LLM-Based Document Q&A with RAG")

# Sidebar for settings
st.sidebar.title("Settings")

# User inputs API keys
groq_api_key = st.sidebar.text_input("Enter the GROQ API Key:", type="password")
openai_api_key = st.sidebar.text_input("Enter the OPENAI API Key:", type="password")

# Select Groq model
models_list = ["gemma2-9b-it", "llama-3.1-8b-instant", "mistral-saba-24b", "qwen2.5-32b"]
model_name = st.sidebar.selectbox("Choose the GROQ Model:", models_list)

if groq_api_key and openai_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Initialize LLM and embeddings
    llm = ChatGroq(model=model_name)
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)

    prompt_template = ChatPromptTemplate.from_template(
        """
        Answer the question based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Question: {input}
        """
    )


    def create_vectors_embeddings():
        if "vectors" not in st.session_state:
            st.session_state.loader = PyPDFDirectoryLoader(documents_folder)
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, embeddings)


    # Upload documents
    uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with open(os.path.join(documents_folder, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success("Files uploaded successfully!")

    # Button to create embeddings
    if st.button("Create Embeddings"):
        create_vectors_embeddings()
        st.success("Vector Database is ready")

    # User query
    user_prompt = st.text_input("Enter your query from the research paper...")

    if user_prompt and "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retriever_chain.invoke({"input": user_prompt})
        st.write(f"Response time: {time.process_time() - start_time:.2f} seconds")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write('-' * 10)