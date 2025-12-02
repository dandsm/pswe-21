import os
import textwrap
from typing import List, Dict

import numpy as np
from pypdf import PdfReader
import ollama

# ===== Configuración =====
DOCS_DIR = "docs"
EMBED_MODEL = "nomic-embed-text"   # modelo de embeddings en Ollama
CHAT_MODEL = "llama3.2:3b"         # modelo de chat en Ollama
CHUNK_SIZE = 800                   # tamaño de cada fragmento (caracteres aprox)
TOP_K = 4                          # cantidad de fragmentos a recuperar


# ===== Carga de PDFs y fragmentación =====
def load_pdfs_to_chunks(docs_dir: str) -> List[Dict]:
    """
    Carga todos los PDFs en docs_dir y los divide en fragmentos de texto.
    Cada PDF se asume que es una ley costarricense (o reglamento relacionado).
    """
    chunks = []

    for filename in os.listdir(docs_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(docs_dir, filename)
        print(f"Ley/Documento: {path}")
        reader = PdfReader(path)

        full_text = ""
        for page in reader.pages:
            text = page.extract_text() or ""
            full_text += "\n" + text

        full_text = full_text.strip()
        if not full_text:
            continue

        # Fragmentación simple por caracteres
        for i in range(0, len(full_text), CHUNK_SIZE):
            chunk_text = full_text[i : i + CHUNK_SIZE].strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "id": f"{filename}-{i // CHUNK_SIZE}",
                    "source": filename,  # nombre del archivo = nombre de la ley
                    "text": chunk_text,
                }
            )

    return chunks


# ===== Embeddings y vector store =====
def embed_text(text: str) -> np.ndarray:
    """Obtiene el embedding desde Ollama para un solo texto."""
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    return np.array(resp["embeddings"][0], dtype="float32")


def build_vector_store(chunks: List[Dict]) -> List[Dict]:
    """Agrega un vector de embedding a cada fragmento."""
    for c in chunks:
        c["embedding"] = embed_text(c["text"])
    return chunks


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def retrieve(query: str, store: List[Dict], k: int = TOP_K) -> List[Dict]:
    """Retorna los k fragmentos más similares para la consulta."""
    q_emb = embed_text(query)
    scored = [(cosine_sim(q_emb, c["embedding"]), c) for c in store]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


# ===== Llamada al LLM (RAG jurídico ambiental) =====
def ask_llm(question: str, contexts: List[Dict]) -> str:
    """
    Llama al modelo de Ollama con el contexto de leyes ambientales.
    El modelo debe responder como un consultor en derecho ambiental costarricense.
    """

    # Construimos un contexto legible juntando los fragmentos
    context_text = "\n\n---\n\n".join(
        f"Ley/Documento: {c['source']}\nTexto:\n{c['text']}"
        for c in contexts
    )

    # Prompt en español, rol de consultor jurídico ambiental
    prompt = f"""
Usted es un consultor experto en Derecho Ambiental Costarricense.
Su tarea es responder únicamente con base en las leyes y textos suministrados en el contexto.
Siempre debe:
- Responder en español de Costa Rica, usando el trato de "usted".
- Mencionar el nombre de la ley o documento (usando el campo "Ley/Documento") cuando sea relevante.
- Indicar claramente si no encuentra información suficiente en el contexto.
- Incluir una nota breve indicando que esto no sustituye la asesoría de un abogado colegiado ni la consulta de la normativa oficial actualizada.

Contexto (extractos de leyes y normativa ambiental de Costa Rica):
{context_text}

Pregunta de la persona usuaria:
{question}

Responda de forma clara y estructurada. Si no existe suficiente base en el contexto, diga que no puede afirmar con certeza y recomiende revisar la ley completa u obtener asesoría legal profesional.
"""

    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    return resp["message"]["content"]


# ===== Bucle principal (CLI) =====
def main():
    if not os.path.isdir(DOCS_DIR):
        raise SystemExit(
            f"No se encontró la carpeta '{DOCS_DIR}'. "
            f"Créela y agregue sus leyes ambientales en PDF."
        )

    print("Cargando leyes ambientales (PDF) y construyendo el índice vectorial...")
    chunks = load_pdfs_to_chunks(DOCS_DIR)
    if not chunks:
        raise SystemExit(
            "No se encontró texto en los PDFs. "
            "Verifique que las leyes sean PDFs con texto (no solo imágenes)."
        )

    vector_store = build_vector_store(chunks)
    print(f"Se indexaron {len(vector_store)} fragmentos de sus leyes.\n")

    print("Consultor de Derecho Ambiental Costarricense listo.")
    print("Escriba su consulta jurídica (o 'salir' para terminar).")
    print()

    while True:
        try:
            q = input("Pregunta legal ambiental: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if q.lower() in {"exit", "quit"}:
            print("Hasta luego.")
            break

        top_chunks = retrieve(q, vector_store, k=TOP_K)
        answer = ask_llm(q, top_chunks)

        print("\n=== Respuesta del consultor ambiental ===\n")
        print(textwrap.fill(answer, width=100))
        print("\n" + "=" * 100 + "\n")


if __name__ == "__main__":
    main()
