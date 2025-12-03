# PoC - AGENTE PARA CONSULTAS LEGALES EN LEGISLACIÓN AMBIENTAL COSTARRICENSE MEDIANTE RAG

## 1. Información general

- **Título del proyecto:** Aprendizaje automático distribuido para consultas legales en legislación ambiental costarricense mediante RAG.
- **Estudiantes:**  
  - Andres Solano Monge  
  - Luis David Chavarria  
  - Gianfranco Bagnarello Hernandez
- **Curso:** PSWE-21 Inteligencia Artificial Distribuida
- **Carrera / Programa:** Maestría en Ingeniería del Software
- **Universidad:** Universidad Cenfotec

---

## 2. Resumen del proyecto

Este proyecto implementa un **agente consultor legal** para **Derecho Ambiental Costarricense**, utilizando un enfoque de **Retrieval-Augmented Generation (RAG)** y **Large Language Models (LLMs)** ejecutados **localmente**.

La prueba de concepto (PoC) se enfoca en:

- Cargar leyes y normativa ambiental de Costa Rica (por ejemplo, **Ley Orgánica del Ambiente**, **Ley de Aguas**, decretos relacionados).
- Vectorizar los textos legales y almacenarlos en un índice interno.
- Permitir consultas en lenguaje natural (en español) sobre temas ambientales.
- Generar respuestas contextuales usando únicamente los fragmentos de ley recuperados por RAG.
- Evaluar el desempeño de varios modelos ligeros (3B–4B parámetros) ejecutados localmente.

---

## 3. Alcance de la PoC

- **Incluye:**
  - Carga y procesamiento de leyes ambientales en formato PDF/HTML.
  - Indexación mediante embeddings y búsqueda semántica.
  - Interfaz de línea de comandos (CLI) para realizar preguntas legales.
  - Evaluación básica de rendimiento (tiempo, tokens, velocidad de generación).

- **No incluye (por ahora):**
  - Validación jurídica formal de las respuestas.
  - Gestión de versiones oficiales del diario La Gaceta.
  - Interfaz web gráfica.
  - Razonamiento jurídico complejo (precedentes, jurisprudencia, etc.).

---

## 4. Arquitectura de la solución

La arquitectura lógica de la PoC se puede resumir en los siguientes componentes:

1. **Fuente de datos legales**  
   - Textos de leyes y reglamentos ambientales costarricenses obtenidos de  
     **[https://pgrweb.go.cr](https://pgrweb.go.cr)** (sitio oficial del Estado).  
   - Formatos utilizados: HTML (copiado a texto) y PDF.

2. **Preprocesamiento y vectorización**  
   - Conversión de PDF → texto (extracción de texto plano).  
   - Fragmentación en **chunks** de tamaño fijo (por ejemplo, ~800 caracteres).  
   - Cálculo de embeddings con un modelo de texto (p. ej., `nomic-embed-text` via Ollama).  
   - Almacenamiento del vector store en caché (archivo `vector_store_cache.pkl`).

3. **Capa RAG**  
   - Dado una pregunta del usuario:
     - Se calcula el embedding de la pregunta.
     - Se recuperan los fragmentos más similares (top-k) del vector store.
     - Se construye un prompt con contexto legal + pregunta.

4. **Modelo de lenguaje local (LLM)**  
   - Ejecución local vía **Ollama** con modelos compactos:
     - `llama3.2:3b`
     - `phi3:3.8b`
     - `qwen2.5:3b`
   - El modelo genera la respuesta en español de Costa Rica, haciendo referencia a las leyes utilizadas como contexto.

5. **CLI / Scripts**  
   - `olama_rag.py`: Consultor legal interactivo.  
   - `olama_rag_with_caching.py`: Consultor legal interactivo con vector store en caché.  
   - `olama_rag_benchmarks.py`: Script de benchmark para comparar modelos.

---

## 6. Requisitos previos

- **Sistema operativo:** GPU +8GB vRAM.
- **Herramientas:**
  - [Ollama](https://ollama.com/) instalado.
  - Python 3.9+ (recomendado).
- **Dependencias Python:**
  - `ollama`
  - `pypdf`
  - `numpy`

Ejemplo de instalación:

```bash
# Crear y activar entorno virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # en macOS/Linux

# Instalar dependencias
pip install ollama pypdf numpy
```


## 7. Datos legales: obtención y características

Las leyes de Costa Rica presentan varias características que facilitan su uso con RAG:

- Generalmente están **disponibles en texto plano** o PDF con texto, lo que simplifica la extracción.
- La mayoría de normativa vigente se puede consultar públicamente en:
  - **https://pgrweb.go.cr** (Procuraduría General de la República).  
- La estructura de las leyes suele ser relativamente **lineal (artículos, capítulos, secciones)**, sin formatos complejos.

### Pasos sugeridos para preparar los datos

1. Ingresar a https://pgrweb.go.cr.
2. Buscar las leyes ambientales de interés (por ejemplo, “Ley Orgánica del Ambiente”, “Ley de Aguas”).
3. Descargar los textos en **PDF**.  
4. Guardar los documentos en la carpeta:

```bash
docs/
  Ley_Organica_del_Ambiente.pdf
  Ley_de_Aguas.pdf
  Decreto_RECOPE_XXXX.pdf
  ...
```

> Nota: Aunque el acceso es público, siempre es recomendable verificar la versión oficial y vigente de cada norma.

---

## 8 Pasos para ejecutar el consultor legal (RAG)

### 8.1. Instalar y preparar modelos en Ollama

[Ollama](https://ollama.com)

```bash
# Instalar Ollama (si no está instalado)

# Iniciar el servicio
ollama serve

# Descargar el modelo de embeddings
ollama pull nomic-embed-text

# Descargar uno o varios modelos de lenguaje a evaluar
ollama pull llama3.2:3b
ollama pull phi3:3.8b
ollama pull qwen2.5:3b
```

### 8.2. Ejecutar el consultor legal interactivo

El script principal (por ejemplo, `olama_rag_with_caching.py`) realiza:

- Carga de PDFs en `docs/`.
- Construcción del vector store (o carga desde caché).
- Bucle de preguntas y respuestas.

Ejemplo:

```bash
python olama_rag_with_caching.py
```

Salida esperada (ejemplo):

```text
Cargando leyes ambientales (PDF) y construyendo el índice vectorial...
Se indexaron XXX fragmentos de sus leyes.

Consultor de Derecho Ambiental Costarricense listo.
Escriba su consulta jurídica (o 'salir' para terminar).

Pregunta legal ambiental: ¿Cuáles son las posibles sanciones según la Ley de Aguas?
...
```

---

## 9. Pasos para ejecutar los benchmarks de modelos

El script de benchmark (por ejemplo, `olama_rag_benchmarks.py`) ejecuta una batería de preguntas sobre varios modelos y produce:

- Resultados por corrida en CSV.
- Resumen por modelo con promedios.

Ejemplo:

```bash
python olama_rag_benchmarks.py
```

Genera salidas similares a:

```text
=========== RESUMEN POR MODELO (PROMEDIOS) ===========

Modelo | N_preguntas | mean_wall_s | mean_total_s | mean_eval_tokens | mean_tokens_per_s
----------------------------------------------------------------------------------------
llama3.2:3b | 3 | 10.358 | 10.344 | 170.33 | 30.21
phi3:3.8b   | 3 | 14.761 | 14.726 | 231.67 | 25.07
qwen2.5:3b  | 3 | 13.035 | 13.031 | 238.00 | 30.34
```

---

## 10. Resultados experimentales

A continuación se muestran resultados de ejemplo obtenidos en una Mac con chip M2 y 8 GB de RAM, para tres preguntas de prueba sobre la Ley de Aguas y decretos relacionados:

### 10.1. Resumen por modelo (promedios)

| Modelo      | N_preguntas | mean_wall_s | mean_total_s | mean_eval_tokens | mean_tokens_per_s | mean_overlap_coverage |
|-------------|-------------|-------------|--------------|------------------|-------------------|-----------------------|
| llama3.2:3b | 3           | 10.358      | 10.344       | 170.33           | 30.21             | 0.819                 |
| phi3:3.8b   | 3           | 14.761      | 14.726       | 231.67           | 25.07             | 0.786                 |
| qwen2.5:3b  | 3           | 13.035      | 13.031       | 238.00           | 30.34             | 0.752                 |


- `llama3.2:3b` presenta la **menor latencia promedio** (menor `mean_wall_s`).
- `qwen2.5:3b` logra la **mayor velocidad de generación** (`mean_tokens_per_s`) y respuestas algo más largas.
- `phi3:3.8b` es el modelo más lento en este escenario, con menor velocidad de tokens por segundo.

---

## 11. Consideraciones legales y éticas

- El agente se basa en texto legal y puede ofrecer **resúmenes y explicaciones**, pero:
  - **No sustituye el criterio profesional de un abogado**.
  - No garantiza la actualización de la normativa (es responsabilidad del usuario actualizar las fuentes).
  - Cualquier decisión legal o administrativa debe sustentarse en la versión oficial de las leyes y, preferiblemente, asesoría jurídica profesional.

- Para documentación y publicaciones, se recomienda **dejar explícito**:
  - La fecha en que se descargaron las leyes.
  - Las versiones / reformas consideradas.
  - Las limitaciones de la herramienta como PoC.

---

## 12. Limitaciones actuales

- Dependencia de la calidad de extracción de texto (PDF → texto).
- Cobertura limitada a las leyes y decretos previamente descargados.
- Evaluación centrada en rendimiento (tiempo, tokens), con evaluación cualitativa limitada.
- Entorno de hardware restringido (8 GB de RAM), lo que limita el tamaño de modelos a 3B–8B parámetros con cuantización.

---


## 13. Referencias

- Sitio oficial de normativa costarricense:  
  - https://pgrweb.go.cr
- Documentación de Ollama:  
  - https://ollama.com
- Biblioteca Python Ollama
  - https://github.com/ollama/ollama-python
- Embedding:
  - https://docs.nomic.ai/platform/embeddings-and-retrieval/text-embedding
