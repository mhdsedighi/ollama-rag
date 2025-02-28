from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader
from tinydb import TinyDB, where
import re, math

records_db = TinyDB('./records-pdf.json')
pdf_ingest_table = records_db.table('pdf_ingest')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=60)

chroma_db = Chroma(
    embedding_function=FastEmbedEmbeddings(),
    persist_directory="./chroma_db_pdfs",
    collection_name="pdfs"
)

def start():
    getFileList("./pdf-files.txt")

def getFileList(filename: str):
    with open(filename) as fp:
        list_size = sum(1 for _ in fp)
        print("Number of PDFs: " + str(list_size))
        fp.close()
    with open(filename) as fp:
        count = 0
        last_pc = 0.0
        for pdf_filename in fp:
            cleaned_pdf_filename = pdf_filename.replace("\n", "")
            percentage = (float(count) / float(list_size) * 100)
            if math.floor(last_pc) < math.floor(percentage):
                print("\n*** {:.2f}% ***\n".format(percentage))
            last_pc = percentage
            print("\n{:.2f}% : ".format(percentage) + cleaned_pdf_filename)
            processPdf(cleaned_pdf_filename)
            count += 1

def processPdf(filename: str):
    # get PDF as chunks
    texts = list(map(lambda c: c.page_content, getPdfChromaDbChunks(filename)))
    print("\t num chunks: " + str(len(texts)))
    # print all extracted text
    print("\t Extracted text:")
    for i, text in enumerate(texts):
        print(f"\t Chunk {i + 1}:")
        print(f"\t {text}")
        print("\t" + "-"*50)
    # get metadata for PDF
    title = getPdfTitle(filename)
    print("\t title: " + title)
    if len(texts) > 0:
        # create Chroma DB metadata for each chunk
        metadatas = []
        for _ in range(len(texts)):
            metadatas.append({"title": title})
        # add data to Chroma DB
        chroma_db.add_texts(
            texts = texts,
            metadatas = metadatas
        )
    else:
        print("\tnothing to ingest, recording for skip anyway")
    # add record to lightweight records DB
    pdf_ingest_table.insert({
        "file": filename,
        "title": title
    })

def getPdfChromaDbChunks(filename: str):
    try:
        docs = PyPDFLoader(file_path=filename).load()
        chunks = text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        return chunks
    except Exception as e:
        print(e)
        return []

def getPdfTitle(filename: str):
    title = ""
    filename_parts = filename.split("/")
    if len(filename_parts) > 0:
        title = filename_parts[-1]
        title = title.replace(".pdf", "")
        title = title.replace(".PDF", "")
    return title

def removeNonAlphaNumOrSpace(s: str):
    return re.sub(r'[^A-Za-z0-9,:\. ]+', '', s)

if __name__ == "__main__":
    start()