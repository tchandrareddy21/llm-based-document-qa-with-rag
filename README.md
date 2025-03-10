# LLM-Based Document Q&A with RAG

## 📌 Overview
- This project implements a Retrieval-Augmented Generation (RAG) system that enables users to upload PDF documents and interact with an LLM to extract meaningful insights. The system uses Groq for querying the LLM and OpenAI for generating embeddings, ensuring efficient document retrieval and accurate answers.

## Architecture
![Architecture](flowcharts/Document%20Q&A%20Process%20Flow.png)
## 🚀 Features
- 📄 **PDF Upload**: Users can upload research papers or other documents in PDF format.
- 🧠 **Embeddings Creation**: Generates vector embeddings for uploaded documents using OpenAI embeddings.
- 💬 **LLM Chat Interface**: A user-friendly chat system that allows interactive Q&A.
- 📌 **Multiple Model Selection**: Users can choose from different Groq models.
- ⚡ **Efficient Search & Retrieval**: Uses FAISS for fast similarity search and document retrieval.

## 🛠 Technologies Used
- **Streamlit** - Web interface for user interaction.
- **LangChain** - Framework for managing retrieval and LLM integration.
- **Groq** - LLM for answering user queries.
- **OpenAI** - Embedding model for document vectorization.
- **FAISS** - High-performance similarity search for document retrieval.
- **PyPDFLoader** - Extracts text from PDF documents.

## 🔧 Installation 
### To run locally follow below steps:
#### Step 1: Clone the repository:
```pycon
git clone https://github.com/tchandrareddy21/llm-based-document-qa-with-rag.git
```
#### Step 2: Create a virtual environment and activate it:
```pycon
conda create -n [env-name] python=3.11 -y
conda actuvate [env-name]
```
#### Step 3: Install dependencies
```pycon
pip install -r requirements.txt
```

#### Step 4: Run the application:
```pycon
streamlit run app.py
```

#### To use live streamlit app 
- Go to the below URL and follow the Usage Guide below:

[LLM Based Document Q&A with RAG - Live APP](https://llm-based-document-q-a-with-rag.streamlit.app/)

## 🎯 Usage Guide
- **Enter API Keys** : Provide your Groq API Key and OpenAI API Key in the sidebar.
- **Select LLM Model** : Choose from the available Groq models.
- **Upload PDFs** : Upload research papers or documents for analysis.
- **Generate Embeddings** : Click the "Create Embeddings" button to process documents.
- **Chat with the AI** : Ask questions and get responses based on document content.

## 📸 Project Screenshots
- Home page
![Home page](screenshots/Home%20page.png)
- After adding API keys
![after API keys added UI](screenshots/after%20API%20keys%20added%20UI.png)
- File uploaded scuccessfully
![File uploaded scuccessfully](screenshots/File%20uploaded%20scuccessfully.png)
- Creating Embeddings
![Creating Embeddings](screenshots/Creating%20Embeddings.png)
- Embeddinga are stored in Vector DB
![Embeddinga are stored in Vector DB](screenshots/Embeddinga%20are%20stored%20in%20Vector%20DB.png)
- Q&A Output
![Q&A Output](screenshots/Q&A%20Output.png)

## 📝 License
This project is licensed under the **MIT License**.
