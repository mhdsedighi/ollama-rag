from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from tinydb import TinyDB
import chromadb
import re, os

# Initialize databases and tools
records_db = TinyDB('./records-pdf.json')
pdf_ingest_table = records_db.table('pdf_ingest')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)

# Modern Chroma client setup with explicit settings
client_settings = chromadb.Settings(
    is_persistent=True,
    persist_directory="./chroma_db_pdfs",
    allow_reset=True
)
client = chromadb.PersistentClient(settings=client_settings)
chroma_db = Chroma(
    client=client,
    collection_name="pdfs",
    embedding_function=FastEmbedEmbeddings()
)

def start():
    get_file_list("./pdf-files.txt")

def get_file_list(filename: str):
    """Process PDFs listed in a file."""
    with open(filename, 'r') as fp:
        pdf_files = [line.strip() for line in fp if line.strip()]
    list_size = len(pdf_files)
    if list_size == 0:
        print("No PDFs found in list.")
        return
    print(f"Number of PDFs: {list_size}")
    
    for count, pdf_filename in enumerate(pdf_files, 1):
        percentage = (count / list_size) * 100
        print(f"\n{percentage:.2f}% : {pdf_filename}")
        process_pdf(pdf_filename, count, list_size)

def get_pdf_author(filename: str):
    try:
        docs = PyPDFLoader(file_path=filename).load()
        for doc in docs[:5]:
            # Look for names after title or DOI, before affiliation
            match = re.search(r"Cross-Cultural Research / [^\n]+\n(.+?)\n(?:University|McMaster)", doc.page_content, re.DOTALL)
            if match:
                authors = match.group(1).strip()
                # Clean up "Sosis, Bressler / COMMUNE LONGEVITY" if present
                return re.sub(r"\s*/\s*COMMUNE LONGEVITY", "", authors)
            # Fallback to metadata
            reader = PdfReader(filename)
            info = reader.metadata
            if info and '/Author' in info:
                return info['/Author']
        return "Unknown"
    except Exception as e:
        print(f"\tError extracting author from {filename}: {e}")
        return "Unknown"

def process_pdf(filename: str, count: int, total: int):
    """Process a single PDF and ingest it into Chroma."""
    if pdf_ingest_table.contains(doc_id=filename):
        print(f"\t{filename} already ingested, skipping.")
        return

    chunks = get_pdf_chroma_db_chunks(filename)
    print(f"\tNum chunks: {len(chunks)}")
    
    if chunks:
        title = get_pdf_title(filename)
        author = get_pdf_author(filename)  # Extract author explicitly
        print(f"\tTitle: {title}, Author: {author}")
        
        metadatas = [{"title": title, "author": author} for _ in range(len(chunks))]
        chroma_db.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=metadatas
        )
        print(f"\tIngested {len(chunks)} chunks into Chroma.")
    else:
        print("\tNothing to ingest, recording for skip.")
    pdf_ingest_table.insert({"file": filename, "title": title, "author": author, "doc_id": filename})

def get_pdf_chroma_db_chunks(filename: str):
    """Load and split a PDF into chunks."""
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"PDF not found: {filename}")
        docs = PyPDFLoader(file_path=filename).load()
        chunks = text_splitter.split_documents(docs)
        return filter_complex_metadata(chunks)
    except Exception as e:
        print(f"\tError processing {filename}: {e}")
        return []

def get_pdf_title(filename: str):
    """Extract a title from the filename."""
    title = os.path.basename(filename)
    return re.sub(r'\.pdf$', '', title, flags=re.IGNORECASE)

if __name__ == "__main__":
    start()