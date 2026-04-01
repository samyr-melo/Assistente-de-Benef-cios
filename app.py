import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, ChatBedrock
import boto3
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Permite que o Front-end fale com o Back-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de dados para a pergunta
class PerguntaInput(BaseModel):
    texto: str

# Inicialização dos modelos AWS
client = boto3.client(service_name="bedrock-runtime", region_name=os.getenv("AWS_REGION"))
embeddings = BedrockEmbeddings(client=client, model_id="amazon.titan-embed-text-v2:0")
llm = ChatBedrock(model_id="meta.llama3-8b-instruct-v1:0", client=client)

# Endpoint para perguntar
@app.post("/perguntar")
async def perguntar(input: PerguntaInput):
    # 1. Carrega o banco FAISS
    db = FAISS.load_local("database_faiss", embeddings, allow_dangerous_deserialization=True)
    # recebe o carregamento local do DB FAISS, passando como paramentro a pasta onde estão os dados do DB, o modelo de conversão matematica e a segurança de integração do langchain com o DB, se não fosse por isso ele não autorizava a abrir o arquivo por precaução. 
    
    # 2. Busca o contexto
    docs = db.similarity_search(input.texto, k=3)
    contexto = "\n".join([doc.page_content for doc in docs])
    
    # 3. Gera a resposta com o Bedrock
    prompt = f"Use o contexto abaixo para responder: {contexto}\n\nPergunta: {input.texto}"
    resposta = llm.invoke(prompt)
    
    return {"resposta": resposta.content}