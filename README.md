# Hybrid Math Reading Assistant

A lightweight **RAG-based reading assistant** for academic papers and research notes.

This project builds a local **document library** from papers/notes, performs **semantic retrieval**, and generates answers with three response modes:

- **Library-grounded**
- **Hybrid**
- **General**

It is designed to help users read research papers more effectively by combining:

- document-aware question answering
- transparent retrieved source snippets
- configurable answer styles
- optional plain-language explanation
- optional simple examples

Although the current version was first built for mathematical papers, the same pipeline can be reused for other domains by replacing the document library.

---

## What this project can do

This assistant can:

- ingest papers and notes from a local `data/` folder
- build a local **vector database**
- answer questions based on retrieved document chunks
- distinguish between:
  - **Library-grounded** answers
  - **Hybrid** answers
  - **General** answers when library coverage is weak
- display retrieved source snippets for transparency
- provide answers in different styles:
  - **Technical**
  - **Plain-language**
- optionally include a **simple example**

### Example use cases

- reading mathematical papers
- understanding definitions and concepts
- getting a plain-language explanation of a technical topic
- asking where a concept is discussed in the current library
- using the same framework with papers from other fields such as physics, biology, or other research areas

---

## Why this project exists

A plain RAG system can still produce answers that sound confident even when retrieval is weak.

This project adds a simple **routing layer** so that the system can better distinguish between:

- questions well supported by the document library
- questions partially supported by the library
- questions outside the current library coverage

The goal is to make the assistant more transparent, more reliable, and more useful for research reading.

---

## Main features

### 1. Document ingestion

Supports local document ingestion from the `data/` folder.

Current supported file types:

- `.pdf`
- `.txt`
- `.md`
- `.tex`

### 2. Semantic retrieval

Uses **Sentence Transformers** embeddings and **ChromaDB** to retrieve the most relevant chunks for a user query.

### 3. Answer generation

Uses an LLM API to generate answers from retrieved context.

### 4. Routing layer

Each answer is assigned one of three modes:

- **Library-grounded**  
  Retrieval is strong, so the answer should rely mainly on the document library.

- **Hybrid**  
  Retrieval is moderately relevant, so the answer may combine library evidence with limited general explanation.

- **General**  
  Retrieval is weak, so the system explicitly states that library coverage is insufficient.

### 5. Retrieved source snippets

The app displays retrieved chunks with:

- source file name
- chunk index
- retrieval distance
- snippet preview

This helps with transparency and basic failure analysis.

### 6. Answer style control

Users can choose:

- **Technical**
- **Plain-language**

### 7. Optional simple example

Users can choose whether to include a short simple example or intuitive illustration in the answer.

---

## Current architecture

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
ChromaDB (Vector Database)
   ↓
Retriever
   ↓
Router (Library-grounded / Hybrid / General)
   ↓
LLM Answer Generation
   ↓
Streamlit UI
···

## Tech stack
Python
Sentence Transformers
ChromaDB
Streamlit
DeepSeek API (OpenAI-compatible client)
PyPDF
Project structure
hybrid-math-reading-assistant/
  data/
  chroma_db/
  app.py
  ingest.py
  retriever.py
  answerer.py
  router.py
  requirements.txt
  .env
File overview
data/
Stores the local document library.
chroma_db/
Stores the local vector index after ingestion.
ingest.py
Loads documents, chunks them, generates embeddings, and stores them in ChromaDB.
retriever.py
Retrieves top-k relevant chunks for a query.
router.py
Assigns one of the three answer modes based on retrieval strength.
answerer.py
Builds context from retrieved chunks and generates the final answer using the LLM API.
app.py
Provides the Streamlit interface.
.env
Stores API credentials.
Setup
1. Clone the repository
git clone <your-repo-url>
cd hybrid-math-reading-assistant
2. Create a virtual environment

On Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
3. Configure your API key

Create a .env file in the project root:

DEEPSEEK_API_KEY=your_api_key_here
4. Add documents

Put your papers / notes into the data/ folder.

How to use
Step 1: Build the document index
.\.venv\Scripts\python.exe ingest.py

This will:

load documents from data/
chunk them
generate embeddings
store them in chroma_db/
Step 2: Launch the app
.\.venv\Scripts\python.exe -m streamlit run app.py

Then open the local URL shown in the terminal, usually:

http://localhost:8501
Step 3: Ask questions

You can ask questions such as:

What is exchangeability?
What is the Chinese Restaurant Process?
exchangeable怎么定义的
Explain this concept in plain language

You can also choose:

Answer style
Technical
Plain-language
whether to Include a simple example
Switching to a new document domain

This project is designed to be reusable across different document collections.

For example, you can replace the current papers with:

number theory papers
graph theory papers
biology papers
physics papers
your own lecture notes or thesis drafts
To switch to a new document library
Replace the files in data/
Delete the old chroma_db/
Re-run:
.\.venv\Scripts\python.exe ingest.py
Launch the app again:
.\.venv\Scripts\python.exe -m streamlit run app.py
Example workflow
Continue querying with the current library
.\.venv\Scripts\python.exe -m streamlit run app.py
Rebuild after changing the library
.\.venv\Scripts\python.exe ingest.py
.\.venv\Scripts\python.exe -m streamlit run app.py
Current limitations
chunking is currently simple character-based chunking
routing is based on hand-tuned distance thresholds
no reranker is used yet
no full evaluation pipeline yet
PDF extraction quality may vary depending on document format
formula-heavy papers may still need better formatting and chunking strategies
text-based academic documents work best in the current version
Planned improvements

Possible next steps include:

better chunking strategies
reranking
document-level filtering
evaluation sheet / failure logging
improved math formatting
question history
local model backend option
better support for cross-domain academic libraries
Status

This is an MVP / early v1.1 version of the project.

Current supported capabilities include:

local document ingestion
semantic retrieval
answer generation
routing-aware answering
retrieved snippet display
technical vs plain-language answer styles
optional simple examples
bilingual usage with English / Chinese queries