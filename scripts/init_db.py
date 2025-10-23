import psycopg
from pgvector.psycopg import register_vector
from dotenv import load_dotenv
import os


load_dotenv()
DSN = os.getenv("PG_DSN")


DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS docs (
id SERIAL PRIMARY KEY,
uri TEXT NOT NULL,
chunk_id INT NOT NULL,
content TEXT NOT NULL,
embedding vector(1536),
meta JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_docs_embedding ON docs USING ivfflat (embedding vector_cosine) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_docs_uri ON docs (uri);
"""


with psycopg.connect(DSN) as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(DDL)
        conn.commit()
print("OK: extensão + tabela criadas")