import os, glob, json, re
from dotenv import load_dotenv
from pypdf import PdfReader
from markdown_it import MarkdownIt
import psycopg
from pgvector.psycopg import register_vector
from openai import OpenAI


load_dotenv()
DSN = os.getenv("PG_DSN")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))


client = OpenAI()
md = MarkdownIt()


def read_text(path:str)->str:
    if path.lower().endswith(".pdf"):
        reader = PdfReader(path)
        return "\n\n".join([p.extract_text() or "" for p in reader.pages])
    elif path.lower().endswith(('.md', '.markdown')):
        return md.render(open(path, 'r', encoding='utf-8').read())
    else:
        return open(path, 'r', encoding='utf-8').read()


def chunk_text(t:str, size:int, overlap:int):
    t = re.sub(r"\s+", " ", t).strip()
    chunks = []
    start = 0
    while start < len(t):
        end = min(len(t), start + size)
        chunks.append(t[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c.strip()]


paths = glob.glob("data/**/*", recursive=True)
paths = [p for p in paths if os.path.isfile(p) and p.split('.')[-1].lower() in ['pdf','md','txt']]
print(f"Arquivos: {len(paths)}")


with psycopg.connect(DSN) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        for path in paths:
            uri = os.path.abspath(path)
            text = read_text(path)
            chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
            print(f"{os.path.basename(path)} → {len(chunks)} chunks")
            if not chunks:
                continue
            # Embeddings em lotes simples
            for i, chunk in enumerate(chunks):
                emb = client.embeddings.create(model=EMBED_MODEL, input=chunk).data[0].embedding
                cur.execute(
                """
                INSERT INTO docs (uri, chunk_id, content, embedding, meta)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (uri, i, chunk, emb, json.dumps({}))
            )
    conn.commit()
print("OK: ingestão concluída")