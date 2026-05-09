# Sherlock Holmes Graph RAG

Sistema de Graph RAG (Retrieval-Augmented Generation con Grafos de Conocimiento) sobre el canon literario de Sherlock Holmes de Sir Arthur Conan Doyle.

**Inteligencia Artificial en Aplicaciones Culturales — Curso 2025-2026**

---

## Descripción

Proyecto académico que construye un sistema Graph RAG capaz de responder preguntas sobre el universo literario de Sherlock Holmes combinando datos no estructurados (texto narrativo) con datos estructurados almacenados en un grafo de conocimiento.

El corpus comprende las 56 historias cortas y las 4 novelas del canon de Conan Doyle, todas de dominio público y disponibles en Project Gutenberg. Los textos se procesan automáticamente para extraer personajes, lugares, crímenes, deducciones y relaciones narrativas.

Graph RAG supera las limitaciones del RAG tradicional: mientras que la búsqueda semántica pura solo recupera fragmentos similares al texto de la consulta, el grafo permite responder preguntas que requieren agregación, conteo, filtrado o razonamiento multisalto (por ejemplo, "¿qué personajes aparecen en más de tres relatos junto a Lestrade?").

El sistema integra tres modos de recuperación: semántico (embeddings sobre chunks), estructurado (consultas Cypher generadas desde lenguaje natural) e híbrido (combinación de ambos), orquestados por un agente router que selecciona la estrategia adecuada a cada consulta.

---

##  Arquitectura

El pipeline se articula en cinco etapas:

1. **Ingesta:** Descarga automática desde Project Gutenberg, separación del texto completo en relatos individuales y chunking consciente de la estructura narrativa (respetando escenas y párrafos).

2. **Extracción:** Sliding context con resumen acumulativo para mantener coherencia entre chunks; extracción multipaso de entidades y relaciones con Gemini; entity resolution para unificar variantes de nombres.

3. **Grafo:** Carga en Neo4j con nodos tipados y relaciones semánticas. Se crean índices vectoriales sobre los nodos `Chunk` y `Character` para habilitar búsqueda híbrida.

4. **Retrieval:** Tres retrievers independientes — vectorial (embeddings + fulltext), Text2Cypher (Gemini genera la consulta Cypher a partir de la pregunta en lenguaje natural) y consultas manuales parametrizadas para patrones recurrentes.

5. **Agentes:** Router que analiza la consulta, selecciona y combina retrievers, descompone preguntas complejas en subpreguntas y aplica un agente crítico para validar y refinar la respuesta final.

---

## Modelo de datos

```cypher
// Nodos principales
(:Story {title, collection, gutenberg_id})
(:Character {name, description, aliases})
(:Location {name, type, description})
(:Crime {type, description, method})
(:Object {name, type, description})
(:Deduction {observation, inference, conclusion, method})
(:Scene {title, description, sequence_order})
(:Event {name, description, sequence_order})
(:Chunk {text, embedding, chunk_index, token_count})

// Relaciones clave
(:Character)-[:APPEARS_IN {role}]->(:Story)
(:Character)-[:KNOWS {relationship_type}]->(:Character)
(:Character)-[:INVESTIGATES {role}]->(:Crime)
(:Deduction)-[:LEADS_TO]->(:Deduction)
(:Scene)-[:FOLLOWS]->(:Scene)
(:Chunk)-[:MENTIONS]->(:Character|Location|Object)
```

Los nodos `Deduction` modelan explícitamente las cadenas de razonamiento deductivo de Holmes como grafos dirigidos, permitiendo reconstruir la lógica de resolución de cada caso. Los nodos `Scene` capturan la estructura narrativa temporal de cada relato y sus relaciones de orden.

---

## Tipos de consultas

**Semánticas** (comprensión del texto):
- "¿Qué papel juega el disfraz en los métodos de investigación de Holmes?"
- "¿Cómo construye Conan Doyle el suspense en The Speckled Band?"
- "¿Qué significado simbólico tienen las cataratas de Reichenbach?"

**Estructuradas** (Cypher sobre el grafo):
- "¿Cuántos casos resuelve Holmes en Londres frente a la campiña?"
- "¿Qué personajes aparecen en más de tres relatos?"
- "Lista los relatos donde estén presentes tanto Lestrade como Watson."

**Híbridas** (combinan grafo y semántica):
- "De los casos en la campiña inglesa, ¿cuáles involucran elementos sobrenaturales?"
- "Entre los relatos con Moriarty, ¿cómo difiere el tono emocional de Holmes?"
- "Para los crímenes que involucran engaño, resume los métodos de los antagonistas."

---

## Estructura del proyecto

```
sherlock-holmes-graph-rag/
├── graphrag/
│   ├── config.py
│   ├── llm/               # Clientes LLM (Gemini + Ollama embeddings)
│   ├── ingestion/         # Descarga, chunking, extracción de entidades
│   ├── graph/             # Gestión de Neo4j
│   ├── retrieval/         # Retrievers (vectorial, Text2Cypher, manual)
│   ├── agents/            # Router agéntico y agente crítico
│   ├── evaluation/        # Evaluación estructurada
│   └── utils/             # Chunking, embeddings
├── notebooks/             # Demos interactivos
├── tests/
├── docs/
├── data/                  # Textos descargados (gitignored)
├── output/                # Resultados intermedios (gitignored)
└── pyproject.toml
```

---

# Instalación

**Prerrequisitos:** Python 3.11+, Neo4j 5.x, Ollama, cuenta Google Cloud con Vertex AI habilitado.

```bash
# Clonar
git clone https://github.com/TU_USUARIO/sherlock-holmes-graph-rag.git
cd sherlock-holmes-graph-rag

# Instalar dependencias con uv
uv sync

# Para dependencias de desarrollo (jupyter, pytest)
uv sync --extra dev

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# Descargar modelo de embeddings
ollama pull nomic-embed-text

# Autenticarse en Google Cloud
gcloud auth application-default login
```

---

## Uso

Los notebooks en `notebooks/` documentan el uso del sistema de forma interactiva:

- `01_ingestion_demo.ipynb` — Construir el grafo de conocimiento desde los textos de Gutenberg.
- `02_retrieval_demo.ipynb` — Probar los tres retrievers de forma aislada y comparada.
- `03_evaluation_demo.ipynb` — Evaluación estructurada del sistema completo.

---

## Hitos

| Hito | Descripción | Fecha |
|------|-------------|-------|
| 1 | Selección de dominio y modelado | 9 marzo 2026 |
| 2 | Construcción del grafo | 13 abril 2026 |
| 3 | Implementación de retrievers | 27 abril 2026 | 
| 4 | Orquestación agéntica | 13 mayo 2026 | 

---

## Stack tecnológico

| Tecnología | Uso | Justificación |
|------------|-----|---------------|
| Gemini 2.5 Flash-Lite | Sliding context, entity resolution | Tareas simples, coste mínimo |
| Gemini 2.5 Flash | Extracción de entidades | Volumen alto, calidad suficiente |
| Gemini 2.5 Pro | Text2Cypher, router, agente crítico | Máxima calidad |
| nomic-embed-text (Ollama) | Embeddings (768 dims) | Local, coste cero |
| Neo4j 5.x | Grafo de conocimiento | Cypher + índices vectoriales nativos |
