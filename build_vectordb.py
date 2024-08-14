from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import pandas as pd
import argparse

#Descargar las dependencias necesarias de nltk
import nltk
nltk.download('punkt')

def build_vectordb(name, chunk_size = 500, chunk_overlap=25):
    
    #PDF LOAD
    loader = PyPDFLoader(f'{name}.pdf')
    documents = loader.load()

    #Splitting
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, language='spanish')
    docs = text_splitter.split_documents(documents)
    
    #generando csv para knowledge base (giskard test)
    df = pd.DataFrame([d.page_content for d in docs], columns=["text"])
    df.to_csv(f'./csv/{name}.csv')

    #generando bd vectorial
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    
    #se guarda la bd en una carpeta en el directorio con el name especificado
    db.save_local(name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', help = 'Nombre del archivo', type=str)
    parser.add_argument('--cs', help = 'Chunk Size', type=int)
    parser.add_argument('--co', help = 'Chunk Overlap', type=int)
    args, optional_args = parser.parse_known_args()
    file_name = args.n
    chunk_size = args.cs
    chunk_overlap = args.co
    if chunk_size and chunk_overlap:
        build_vectordb(name=file_name,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    else:
        build_vectordb(file_name)
