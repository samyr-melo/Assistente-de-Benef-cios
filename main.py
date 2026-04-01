import os
import boto3
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import PyPDF2  # Lembre-se de rodar: pip install PyPDF2

load_dotenv()

PASTA = "documentos"

def carregar_textos_da_pasta(caminho_pasta):
    conteudos = []
    if not os.path.exists(caminho_pasta):
        print(f"Erro: A pasta {caminho_pasta} não foi encontrada.")
        return []

    for arquivo in os.listdir(caminho_pasta):
        caminho_completo = os.path.join(caminho_pasta, arquivo)
        
        # Lógica para TXT
        if arquivo.lower().endswith(".txt"):
            with open(caminho_completo, 'r', encoding='utf-8') as f:
                conteudos.append(f.read())
        
        # Lógica para PDF (Adicionada aqui!)
        elif arquivo.lower().endswith(".pdf"):
            try:
                with open(caminho_completo, 'rb') as f:
                    leitor = PyPDF2.PdfReader(f)
                    texto_pdf = ""
                    for pagina in leitor.pages:
                        texto_pdf += pagina.extract_text() + "\n"
                    if texto_pdf.strip():
                        conteudos.append(texto_pdf)
            except Exception as e:
                print(f"Erro ao ler PDF {arquivo}: {e}")
                
    return conteudos

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv('AWS_REGION'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
)

def get_embedding(text):
    body = json.dumps({"inputText": text})
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("embedding")

# --- EXECUÇÃO PRINCIPAL ---

documentos = carregar_textos_da_pasta(PASTA)

if not documentos:
    print(f"⚠️ Atenção: Nenhum arquivo válido (.txt ou .pdf) encontrado em '{PASTA}'.")
else:
    print(f"✅ Processando {len(documentos)} arquivos...")

    # TUDO ISSO DEVE ESTAR DENTRO DO ELSE
    try:
        # 1. Gera embeddings
        embeddings = [get_embedding(doc) for doc in documentos]
        embeddings_np = np.array(embeddings).astype('float32')

        # 2. Cria o índice FAISS
        dimensao = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimensao)
        index.add(embeddings_np)

        # 3. Salva os arquivos
        faiss.write_index(index, "beneficios.index")
        with open("documentos.json", "w", encoding='utf-8') as f:
            json.dump(documentos, f, ensure_ascii=False)

        print("🚀 Banco vetorial criado com sucesso!")

    except Exception as e:
        print(f"❌ Erro durante a criação do banco: {e}")

# Função de busca (permanece fora pois será chamada depois)
def busca_contexto_faiss(pergunta, top_k=2):
    if not os.path.exists("beneficios.index"):
        return ["Erro: Banco de dados não encontrado."]
        
    index = faiss.read_index("beneficios.index")
    with open("documentos.json", "r", encoding='utf-8') as f:
        docs = json.load(f)

    pergunta_embedding = np.array([get_embedding(pergunta)]).astype('float32')
    distancias, indices = index.search(pergunta_embedding, top_k)
    
    return [docs[i] for i in indices[0] if i != -1]

# Teste final
if os.path.exists("beneficios.index"):
    pergunta_usuario = input("\nEscreva sua pergunta sobre benefícios: ")
    contexto = busca_contexto_faiss(pergunta_usuario)
    print(f"\nTrechos encontrados no banco:\n{contexto}")
