# NLP Assignment 6: Naive RAG vs. Contextual Retrieval
### Comparative Analysis and Implementation for LLM Chapter 7

This repository contains a full-stack RAG (Retrieval-Augmented Generation) application designed to compare **Naive RAG** with **Contextual Retrieval** strategies. The project focuses on improving retrieval accuracy for technical documents by "situating" individual chunks within the broader context of the source material.

---

## 📊 Performance Comparison (Task 2.3)

The evaluation was conducted using **ROUGE** metrics to compare the quality of generated answers against a gold-standard reference from Chapter 7.

| Metric | Naive RAG (Baseline) | Contextual Retrieval (Proposed) | % Improvement |
| :--- | :--- | :--- | :--- |
| **ROUGE-1** | 0.3542 | 0.4821 | **+36.1%** |
| **ROUGE-2** | 0.1210 | 0.2154 | **+78.0%** |
| **ROUGE-L** | 0.3122 | 0.4490 | **+43.8%** |

### 🔍 Analysis of Results
Naive RAG often fails when chunks are semantically similar but contextually different (e.g., distinguishing between different layers of a Transformer). **Contextual Retrieval** solves this by prepending a 1-sentence situational summary to every chunk before embedding. This "Global-to-Local" mapping allows the retriever to achieve significantly higher precision, as evidenced by the **78% jump in ROUGE-2 scores**.

---

## 🤖 Web Application Showcase (Task 3)

The application features a Streamlit-based UI that utilizes **Groq (Llama-3.1-8b)** for generation and **HuggingFace (MiniLM-L6-v2)** for native M4-optimized embeddings.

### 💬 Chat Interface
*The assistant provides technical answers based on Chapter 7 context.*

<img width="730" height="697" alt="Screenshot 2026-03-19 at 9 38 37 pm" src="https://github.com/user-attachments/assets/2a35cbcc-9fb4-4145-ae6e-f680ec21883d" />

<img width="730" height="413" alt="Screenshot 2026-03-19 at 9 37 08 pm" src="https://github.com/user-attachments/assets/611ab309-5847-48f6-9e2f-b645d88286ef" />

### 📚 Source Citations & Metadata (Task 3.4)
*The system explicitly cites the source chunks and page numbers used for every response.*

<img width="1470" height="776" alt="Screenshot 2026-03-19 at 9 36 10 pm" src="https://github.com/user-attachments/assets/37337527-56f3-4c9a-8276-ac2e5477201d" />


---

## 🛠️ Implementation Details

### 1. M4 Mac Optimization
To run this project natively on Apple Silicon (M4), specific environment configurations were used to ensure compatibility with `MPS` (Metal Performance Shaders):
- **Architecture:** `arm64`
- **PyTorch Version:** 2.4+ (Native M4 Support)
- **Environment:** `uv` virtual environment for high-speed dependency resolution.

### 2. Contextual Retrieval Logic (Task 2.2)
Instead of standard chunking, this project implements a **Contextual Prefixing** pipeline:
1. Load the full document (`7.pdf`).
2. Generate a concise context summary for each chunk using the LLM.
3. Prepend the summary: `Context: [Summary] \n\n [Original Chunk]`.
4. Embed the "enriched" chunk into the Chroma vector database.

---

## 🚀 How to Run

### Local (M4 Mac)
```bash
# Initialize and activate venv
arch -arm64 uv venv --python 3.12
source .venv/bin/activate

# Install native dependencies
uv pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
