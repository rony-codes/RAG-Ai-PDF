from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

#loading file in python prgm

loader =  PyPDFLoader(file_path=pdf_path)
docs = loader.load()

print(docs[12])

#Spliting the docs into smaller chunks 
test_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=400
)

chunks = test_splitter.split_documents(documents=docs)

#vector Embeddings
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning_rag"
)
