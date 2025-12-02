import os
import textwrap
import pickle
import time
import statistics as stats
from typing import List, Dict, Any, Optional

import numpy as np
from pypdf import PdfReader
import ollama

# ========= CONFIGURACIÓN GENERAL =========

DOCS_DIR = "docs"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
TOP_K = 4
CACHE_FILE = "vector_store_cache.pkl"

# Modelos a comparar (ajuste a su gusto)
MODELS_TO_TEST = [
    "llama3.2:3b",
    "phi3:3.8b",
    "qwen2.5:3b",
]

# Preguntas de benchmark (ejemplos, puede cambiarlas por su batería real)
TEST_QUESTIONS = [
    "¿Cuáles es el número de Ley de Aguas?",
    "¿Cuáles son los decretos relacionados con Recope en su titulo?",
    "¿Qué dice son las posibles sanciones de la Ley de Aguas?",
]


# ========= UTILIDADES PARA LEYES Y VECTOR STORE =========

def load_pdfs_to_chunks(docs_dir: str) -> List[Dict[str, Any]]:
    """Carga todos los PDFs en docs_dir y los divide en fragmentos de texto."""
    chunks: List[Dict[str, Any]] = []

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

        for i in range(0, len(full_text), CHUNK_SIZE):
            chunk_text = full_text[i: i + CHUNK_SIZE].strip()
            if not chunk_text:
                continue

            chunks.append(
                {
                    "id": f"{filename}-{i // CHUNK_SIZE}",
                    "source": filename,
                    "text": chunk_text,
                }
            )

    return chunks


def embed_text(text: str) -> np.ndarray:
    """Obtiene el embedding desde Ollama para un solo texto."""
    resp = ollama.embed(model=EMBED_MODEL, input=text)
    return np.array(resp["embeddings"][0], dtype="float32")


def build_vector_store(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Agrega un vector de embedding a cada fragmento."""
    for c in chunks:
        c["embedding"] = embed_text(c["text"])
    return chunks


def save_vector_store_to_cache(vector_store: List[Dict[str, Any]], cache_path: str) -> None:
    with open(cache_path, "wb") as f:
        pickle.dump({"vector_store": vector_store}, f)
    print(f"Vector store guardado en caché: {cache_path}")


def load_vector_store_from_cache(cache_path: str) -> List[Dict[str, Any]]:
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    print(f"Vector store cargado desde caché: {cache_path}")
    return data["vector_store"]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def retrieve(query: str, store: List[Dict[str, Any]], k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retorna los k fragmentos más similares para la consulta."""
    q_emb = embed_text(query)
    scored = [(cosine_sim(q_emb, c["embedding"]), c) for c in store]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:k]]


# ========= PROMPT DEL CONSULTOR AMBIENTAL =========

def build_prompt(question: str, contexts: List[Dict[str, Any]]) -> str:
    context_text = "\n\n---\n\n".join(
        f"Ley/Documento: {c['source']}\nTexto:\n{c['text']}"
        for c in contexts
    )

    prompt = f"""
Usted es un consultor experto en Derecho Ambiental Costarricense.
Su tarea es responder únicamente con base en las leyes y textos suministrados en el contexto.
Siempre debe:
- Responder en español de Costa Rica, usando el trato de "usted".
- Mencionar el nombre de la ley o documento (campo "Ley/Documento") cuando sea relevante.
- Indicar claramente si no encuentra información suficiente en el contexto.
- Incluir una nota breve indicando que esto no sustituye la asesoría de un abogado colegiado ni la consulta de la normativa oficial actualizada.

Contexto (extractos de leyes y normativa ambiental de Costa Rica):
{context_text}

Pregunta de la persona usuaria:
{question}

Responda de forma clara y estructurada. Si no existe suficiente base en el contexto, diga que no puede afirmar con certeza y recomiende revisar la ley completa u obtener asesoría legal profesional.
"""
    return prompt.strip()


# ========= LLAMADA AL LLM + MÉTRICAS =========

def answer_with_model(
    model_name: str,
    question: str,
    vector_store: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta una consulta RAG para una pregunta y un modelo, devolviendo
    tanto la respuesta como las métricas de rendimiento de Ollama.
    """
    if options is None:
        # Misma configuración para todos los modelos (IMPORTANTE para el paper)
        options = {
            "temperature": 0.2,   # baja aleatoriedad
            "num_predict": 512,   # límite máximo de tokens de salida
        }

    contexts = retrieve(question, vector_store, k=TOP_K)
    prompt = build_prompt(question, contexts)

    start_wall = time.perf_counter()
    resp = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options=options,
        stream=False,  # necesario para obtener métricas en un solo JSON :contentReference[oaicite:1]{index=1}
    )
    end_wall = time.perf_counter()

    wall_time_s = end_wall - start_wall

    # Métricas de Ollama (pueden venir como atributos o como claves del dict)
    # Lo manejamos de forma robusta.
    def _get_field(obj, key: str, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    total_duration_ns = _get_field(resp, "total_duration", 0) or 0
    eval_count = _get_field(resp, "eval_count", 0) or 0
    eval_duration_ns = _get_field(resp, "eval_duration", 0) or 0
    prompt_eval_count = _get_field(resp, "prompt_eval_count", 0) or 0
    prompt_eval_duration_ns = _get_field(resp, "prompt_eval_duration", 0) or 0

    total_duration_s = total_duration_ns / 1e9
    eval_duration_s = eval_duration_ns / 1e9
    prompt_eval_duration_s = prompt_eval_duration_ns / 1e9

    tokens_per_s = (
        eval_count / eval_duration_s if eval_duration_s > 0 else None
    )  # fórmula recomendada por la doc oficial :contentReference[oaicite:2]{index=2}

    answer_text = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content

    return {
        "model": model_name,
        "question": question,
        "answer": answer_text,
        "wall_time_s": wall_time_s,
        "total_duration_s": total_duration_s,
        "eval_tokens": eval_count,
        "eval_duration_s": eval_duration_s,
        "tokens_per_s": tokens_per_s,
        "prompt_tokens": prompt_eval_count,
        "prompt_eval_duration_s": prompt_eval_duration_s,
    }


# ========= BENCHMARK =========

def warmup_model(model_name: str) -> None:
    """Pequeña llamada de warmup para que cargar el modelo no contamine las mediciones."""
    _ = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": "Warmup only, responda una frase muy corta."}],
        options={"temperature": 0.0, "num_predict": 16},
        stream=False,
    )


def benchmark_models(
    vector_store: List[Dict[str, Any]],
    models: List[str],
    questions: List[str],
) -> List[Dict[str, Any]]:
    """
    Corre todas las preguntas para cada modelo y devuelve una lista de
    resultados detallados (una fila por modelo x pregunta).
    """
    all_results: List[Dict[str, Any]] = []

    for model in models:
        print(f"\n========== Benchmark para modelo: {model} ==========")
        print("Calentando modelo...")
        warmup_model(model)

        for q in questions:
            print(f"  >Pregunta: {q[:80]}...")
            metrics = answer_with_model(model, q, vector_store)
            all_results.append(metrics)

            # Imprime resumen corto por corrida
            print(metrics['answer'].strip().replace("\n", " ")[:200] + "...")
            print(
                f"    wall_time_s={metrics['wall_time_s']:.3f}  "
                f"total_duration_s={metrics['total_duration_s']:.3f}  "
                f"eval_tokens={metrics['eval_tokens']}  "
                f"tokens_per_s="
                f"{metrics['tokens_per_s']:.1f}" if metrics['tokens_per_s'] else "N/A"
            )

    return all_results


def summarize_results(all_results: List[Dict[str, Any]]) -> None:
    """Imprime resumen por modelo y un pseudo-CSV para copiar al artículo."""
    # Agrupar por modelo
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_results:
        by_model.setdefault(r["model"], []).append(r)

    print("\n\n=========== RESUMEN POR MODELO (PROMEDIOS) ===========\n")
    header = (
        "Modelo | N_preguntas | mean_wall_s | mean_total_s | "
        "mean_eval_tokens | mean_tokens_per_s"
    )
    print(header)
    print("-" * len(header))

    for model, rows in by_model.items():
        wall = [r["wall_time_s"] for r in rows]
        total = [r["total_duration_s"] for r in rows]
        eval_tokens = [r["eval_tokens"] for r in rows]
        tps = [r["tokens_per_s"] for r in rows if r["tokens_per_s"] is not None]

        print(
            f"{model} | {len(rows)} | "
            f"{stats.mean(wall):.3f} | "
            f"{stats.mean(total):.3f} | "
            f"{stats.mean(eval_tokens)} | "
            f"{stats.mean(tps) if tps else float('nan')}"
        )

    # También imprimimos en formato CSV para copiar/pegar
    print("\n\n=========== CSV PARA ARTÍCULO (por corrida) ===========\n")
    print(
        "model,question,wall_time_s,total_duration_s,eval_tokens,eval_duration_s,"
        "tokens_per_s,prompt_tokens,prompt_eval_duration_s"
    )
    for r in all_results:
        q_short = r["question"].replace("\n", " ").replace(",", ";")
        print(
            f"{r['model']},{q_short},"
            f"{r['wall_time_s']:.6f},{r['total_duration_s']:.6f},"
            f"{r['eval_tokens']},{r['eval_duration_s']:.6f},"
            f"{r['tokens_per_s']:.6f},"
            f"{r['prompt_tokens']},{r['prompt_eval_duration_s']:.6f}"
        )


# ========= MAIN =========

def main():
    # Cargar o construir el vector store de leyes
    if os.path.exists(CACHE_FILE):
        vector_store = load_vector_store_from_cache(CACHE_FILE)
    else:
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
        print(f"Se indexaron {len(vector_store)} fragmentos de sus leyes.")
        save_vector_store_to_cache(vector_store, CACHE_FILE)

    # Ejecutar benchmark
    results = benchmark_models(vector_store, MODELS_TO_TEST, TEST_QUESTIONS)
    summarize_results(results)


if __name__ == "__main__":
    main()
