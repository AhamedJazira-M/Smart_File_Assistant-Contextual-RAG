
# 🧠 Contextual RAG with Groq, LangChain & Gradio

This project enables you to ask intelligent questions over a custom CSV (`context.csv`) using Groq's blazing-fast **LLaMA 3**, **LangChain**, and **Gradio** UI. It leverages **Contextual Compression Retriever** to provide concise, relevant answers from your documents.

---

## 🔧 Features

- 💬 Query your CSV using natural language
- 🧠 Uses LLaMA 3 via Groq for fast & accurate responses
- 🧱 Contextual Compression with LangChain retrievers
- 🔎 Embeddings via HuggingFace transformers
- 🌐 Simple Gradio interface with sample queries

---

## 📁 Project Structure

├── context.csv # Your knowledge base (upload your own)
├── app.py # Main RAG pipeline + Gradio interface
├── README.md # You're reading it!

## 🚀 How to Run (in Google Colab)

1. Upload your `context.csv` file.
2. Make sure your [Groq API Key](https://console.groq.com/) is set via `userdata.get('Groq_AJ')`
3. Install required libraries:
   ```bash
   !pip install -q langchain groq langchain-groq chromadb datasets pandas gradio
