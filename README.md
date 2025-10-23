# Mini RAG — FastAPI + Postgres/pgvector + OpenAI

> Ingerir documentos, indexar com vetores, recuperar trechos relevantes com KNN e gerar respostas **ancoradas nas fontes**.

## ✨ Visão geral
Este projeto implementa um **Retrieval-Augmented Generation (RAG)**:
1) **Ingestão** de PDFs/MD/TXT em `data/`,  
2) **Chunking + overlap** e geração de **embeddings**,  
3) **Indexação vetorial** no Postgres com **pgvector**,  
4) **Busca por similaridade** (KNN) dos chunks,  
5) **Geração de resposta** por um LLM com **guardrails** (responder apenas com base nas fontes e citar referências).

---

## 🧱 Stack
- **API**: FastAPI
- **Índice vetorial**: Postgres + **pgvector** (Docker)
- **Embeddings & Geração**: OpenAI API (pode ser trocado por endpoint compatível)
- **Cliente HTTP**: `curl` ou qualquer REST client

---

## 📁 Estrutura do projeto
mini-rag/
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── data/ # seus PDFs/MD/TXT (não versionar)
├── scripts/
│ ├── init_db.py # cria extensão pgvector + tabela + índices
│ └── ingest.py # chunking + embeddings + upsert no Postgres
└── app/
├── main.py # FastAPI (/ask, /health)
└── rag.py # retrieve + prompt + geração

---

## 🔧 Requisitos
- Docker e Docker Compose
- Python 3.10+ (recomendado usar `venv`)
- Chave de API válida (OpenAI ou compatível com o SDK)

---

## ⚙️ Configuração
Copie o `.env.example` para `.env` e edite:
