import os, psycopg
from pgvector.psycopg import register_vector
from pydantic import BaseModel
from openai import OpenAI


EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
DSN = os.getenv("PG_DSN")
TOP_K = int(os.getenv("TOP_K", 5))


client = OpenAI()


class Retrieved(BaseModel):
    uri: str
    chunk_id: int
    content: str


class Answer(BaseModel):
    answer: str
    sources: list[Retrieved]


PROMPT = (
"Você é um assistente. Responda APENAS com base nos trechos fornecidos. "
"Se não houver evidência suficiente, diga ‘não sei com base nas fontes’. Ao final, liste as fontes."
)




def embed(q: str):
    return client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding




def retrieve(query: str) -> list[Retrieved]:
    q = embed(query)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT uri, chunk_id, content
                FROM docs
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (q, TOP_K),
            )
            rows = cur.fetchall()
    return [Retrieved(uri=r[0], chunk_id=r[1], content=r[2]) for r in rows]




def generate(query: str, ctx: list[Retrieved]) -> Answer:
    context = "\n\n---\n\n".join([f"[Fonte {i+1}] {c.uri} (chunk {c.chunk_id})\n{c.content}" for i,c in enumerate(ctx)])
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": f"Pergunta: {query}\n\nContexto:\n{context}"}
    ]
    resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages)
    answer = resp.choices[0].message.content
    return Answer(answer=answer, sources=ctx)