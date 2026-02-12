from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

urls = [
    "https://raysofgrace.ac.ug",
    "https://raysofgrace.ac.ug/about",
    "https://raysofgrace.ac.ug/team",
    "https://raysofgrace.ac.ug/programs",
    "https://raysofgrace.ac.ug/admission",
    "https://raysofgrace.ac.ug/blog",
    "https://raysofgrace.ac.ug/document",
    "https://raysofgrace.ac.ug/contact",
]

docs = []
for url in urls:
    loader = WebBaseLoader(url)
    docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vectorstore.persist()

print("Website data indexed successfully.")
