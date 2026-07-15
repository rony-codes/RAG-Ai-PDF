# Chat with your PDF — 100% local RAG

Ask questions about a PDF and get answers grounded in the document itself —
no OpenAI key, no cloud calls. Embeddings and the chat model both run
locally through [Ollama](https://ollama.com), with [Qdrant](https://qdrant.tech)
as the vector database.

## How it works

1. **`index.py`** loads a PDF, splits it into overlapping chunks, turns each
   chunk into a vector using a local embedding model, and stores it in Qdrant.
2. **`chat.py`** takes your question, searches Qdrant for the most relevant
   chunks, and asks a local LLM to answer using *only* that context — citing
   the page number it found the answer on.

```
PDF → split into chunks → embed locally (Ollama) → store in Qdrant
                                                          ↓
your question → search Qdrant for relevant chunks → local LLM answers
```

## Stack

- **PyPDFLoader** — reads the PDF
- **RecursiveCharacterTextSplitter** — splits it into overlapping chunks
- **Ollama (`nomic-embed-text`)** — generates embeddings locally
- **Qdrant** — stores and searches the vectors (via Docker)
- **Ollama (`qwen3:4b`)** — answers questions using retrieved context

## Prerequisites

- Python 3.10+
- [Docker](https://www.docker.com/)
- [Ollama](https://ollama.com) installed and running

Pull the two models this project uses:

```bash
ollama pull nomic-embed-text
ollama pull qwen3:4b
```

## Setup

```bash
git clone https://github.com/rony-codes/RAG-PDF.git
cd <your-repo-folder>
pip install -r requirements.txt
```

Start Qdrant:

```bash
docker compose up -d
```

Drop a PDF named `nodejs.pdf` in the project root (or edit the filename in
`index.py` to point at your own file).

## Usage

**1. Ingest the PDF** — reads it, chunks it, embeds it, and stores it in Qdrant:

```bash
python index.py
```

**2. Ask questions about it:**

```bash
python chat.py
```

```
Ask Something: What does this document say about the event loop?

🤖: According to the document (page 14), the event loop is what allows
Node.js to perform non-blocking I/O operations...
```

## Why local?

- No API costs — everything runs on your own machine
- No data leaves your machine — useful for private or sensitive documents
- Good way to actually understand how RAG works end to end, instead of
  calling a hosted API and treating it as a black box

## Notes

- `chunk_overlap` is currently set high (400 on a 1000 chunk size); a lower
  overlap (~150–200) trades a little context continuity for fewer, cheaper
  embeddings.
- The system prompt in `chat.py` is intentionally strict — it tells the
  model to say *"I could not find this information in the PDF"* rather than
  guess, and to always cite a page number.