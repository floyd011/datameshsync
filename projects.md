# Projects

[Home](index.md) &nbsp; &nbsp; [About](about.md) &nbsp; &nbsp;[Skills](skills.md) &nbsp; &nbsp;[Projects](projects.md) &nbsp; &nbsp;[Experience](expirience.md)

## Project Highlights

### Private RAG Pipeline with FastAPI, Qdrant, and Fine-Tuned LLaMA

**Document Ingestion (PDF Import Module)**:

- Implemented a PDF ingestion pipeline that accepts multiple documents through a drag-and-drop UI (Streamlit) or REST API (FastAPI).
- Each PDF is parsed using PyMuPDF or pdfplumber to extract clean, structured text.
- Metadata (e.g., title, source, page count) is captured for later filtering and display.

**Embedding Generation & Storage**:

- Extracted text chunks (based on token limits) are passed to a selected embedding model (e.g., all-MiniLM-L6-v2 or bge-base-en) using HuggingFace or SentenceTransformers.
- Each chunk is vectorized and stored in Qdrant, a high-speed vector search engine, along with its metadata and original content.
- The vector index is organized by document and optionally namespace for multi-tenant support.

**Predictive Tagging (Keyword Extraction & Classification)**:

- For each chunk, we used KeyBERT or a transformer-based classifier to extract semantic tags.
- Tags are added to the Qdrant metadata fields for advanced filtering and smart querying (e.g., “find security-related documents tagged with ‘OAuth’”).
- An option for human-in-the-loop review of auto-tags is included in the Streamlit UI.

**Backend Search API (FastAPI with RAG Logic)**:

- A RESTful API is built using FastAPI, exposing endpoints for:
- Document upload and indexing
- Vector similarity search with hybrid (tag + semantic) filtering
- Prompt-based document querying (RAG)
- API handles chunked retrieval from Qdrant based on user queries, re-ranks results, and composes context windows for LLM input.

**Local LLM Integration (LLaMA + Ollama)**:

- Deployed a fine-tuned version of LLaMA 3 using LoRA/QLoRA for specialization in   documentation-based search tasks.
- The model is served locally via Ollama, exposing a simple HTTP API.
- Input prompt includes top-k document chunks retrieved from Qdrant + user query → model generates the final answer or summary.
- Training used a custom instruction dataset based on the ingested PDF corpus.

**Streamlit Frontend UI**:

- Developed a clean, user-friendly interface to:
- Upload and preview PDF documents
- View automatically extracted tags
- Enter search queries and view generated answers
- Inspect which chunks were retrieved and used in the generation
- UI is directly connected to FastAPI backend via HTTP calls

### Real-Time Analytics Platform

- **Objective:**  
  Developed a real-time analytics platform to process and analyze data for business intelligence applications.

- **Technologies Used:**  
  Debezium, Kafka, Python, Greenplum, Docker, Kubernetes, Graylog, Prometheus, Grafana

- **Key Achievements:**  
  - Achieved sub-second latency in capturing and processing database changes  
  - Designed and implemented scalable microservices capable of handling millions of events daily  
  - Built Grafana dashboards to monitor system performance and critical operational metrics  

### Microservices-Based Data Pipeline

- **Objective:**  
  Developed a microservices-based architecture to enable real-time streaming and processing of large datasets from multiple relational databases.

- **Technologies Used:**  
  Debezium, Kafka, Python, Docker, Kubernetes

- **Key Achievements:**  
  - Automated deployment and scaling of services using Kubernetes  
  - Reduced operational complexity by centralizing logs through Graylog containers  
  - Improved fault tolerance via Kafka partitioning and replication strategies  

### GPS Tracking & Telemetry Data Pipeline

- Designed and maintained high-performance, real-time data pipelines focused on GPS tracking systems and telemetry data ingestion.  
- Engineered solutions capable of processing hundreds of millions of records per day to support real-time analytics.  
- Leveraged cloud infrastructure, containerization, and data streaming technologies to optimize workflows for high-velocity, high-volume environments.  
