import logging  # Import logging module

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFDirectoryLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

load_dotenv()

# Aquí son los nombres del index que llamamos en pinecone.
index_name = "school-kb"
# el nombre del embedding model que configuramos en python
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# aqui se pueden cargar directorios completos
try:
    logging.info("Loading documents from directory...")
    loader = DirectoryLoader("./data", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)  # Use custom loader
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    readme_content = documents[0].page_content
    # print sample of uploaded documents
    print("SAMPLE OF UPLOADED DOCUMENT",readme_content[:250])
except Exception as e:
    logging.error(f"Error loading documents: {e}")

# declare chunk size for the text splitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# split documents 
try:
    logging.info("Splitting documents...")
    docs = text_splitter.split_documents(documents)
    logging.info(f"Split into {len(docs)} chunks.")
except Exception as e:
    logging.error(f"Error splitting documents: {e}")

try:
    logging.info("Creating vector store from documents...")
    vectorstore_from_docs = PineconeVectorStore.from_documents(
        docs,
        index_name=index_name,
        embedding=embeddings
    )
    logging.info("Vector store created successfully.")
except Exception as e:
    logging.error(f"Error creating vector store: {e}")