# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_aws import BedrockEmbeddings  # Em vez de OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_aws import ChatBedrock
# from dotenv import load_dotenv
# import boto3
# import os

# load_dotenv()
# PASTA = "documentos"
# NOME_BANCO_LOCAL = "database_faiss"

# def get_embeddings_model():
#     client = boto3.client(service_name = "bedrock-runtime", region_name=os.getenv("AWS_REGION"))
#     return BedrockEmbeddings(client=client, model_id="amazon.titan-embed-text-v2:0")

# def carregar_documentos():
#     loader = PyPDFDirectoryLoader(PASTA)
#     return loader.load()

# def dividir_docs(arquivos):
#     separador = RecursiveCharacterTextSplitter(
#         chunk_size = 1000,
#         chunk_overlap = 100,
#         length_function = len,
#         add_start_index = True
#     )
#     chunks = separador.split_documents(arquivos)
#     print(len(chunks))
#     return chunks

# def vetorizar(chunks):
#     embeddings = get_embeddings_model()
#     db = FAISS.from_documents(chunks, embeddings)
#     db.save_local(NOME_BANCO_LOCAL)   
#     print(f"Banco de Dados salvo em: {NOME_BANCO_LOCAL}")

# def create_db():
#     docs = carregar_documentos()
#     if not docs:
#         print("Nenhum documento encontrado na pasta.")
#         return
    
#     chunk = dividir_docs(docs)
#     vetorizar(chunk)


# if __name__ == "__main__":
#     create_db()

import os
import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

PASTA = "documentos"
NOME_BANCO_LOCAL = "database_faiss"

def get_embeddings_model():
    # Garantindo que o cliente pegue as chaves do seu .env
    client = boto3.client(
        service_name="bedrock-runtime", 
        region_name=os.getenv("AWS_REGION"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )
    
    return BedrockEmbeddings(
        client=client, 
        model_id="amazon.titan-embed-text-v2:0"
    )

def carregar_documentos():
    if not os.path.exists(PASTA):
        os.makedirs(PASTA) # Cria a pasta caso não exista
        return []
    
    loader = PyPDFDirectoryLoader(PASTA)
    return loader.load()

def dividir_docs(arquivos):
    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True
    )
    chunks = separador.split_documents(arquivos)
    print(f"✅ {len(chunks)} pedaços de texto gerados.")
    return chunks

def vetorizar(chunks):
    try:
        embeddings = get_embeddings_model()
        print("⏳ Gerando vetores na AWS... isso pode levar alguns segundos.")
        
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(NOME_BANCO_LOCAL) 
        
        print(f"🚀 Sucesso! Banco salvo em: {NOME_BANCO_LOCAL}")
    except Exception as e:
        print(f"❌ Erro na vetorização: {e}")

def create_db():
    docs = carregar_documentos()
    if not docs:
        print(f"⚠️ Nenhum PDF encontrado na pasta '{PASTA}'.")
        return
    
    chunk = dividir_docs(docs)
    vetorizar(chunk)

if __name__ == "__main__":
    create_db()