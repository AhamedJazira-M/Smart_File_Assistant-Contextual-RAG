# 📦 IMPORTS

import os
import pandas as pd
import gradio as gr
from datasets import Dataset

from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# 🔧 ENVIRONMENT SETUP

from google.colab import userdata
os.environ["GROQ_API_KEY"] = userdata.get("Groq_AJ")  #Your Groq API Key here


# 📥 LOAD & PROCESS DOCUMENTS

loader = CSVLoader("/content/context.csv")  # ✅ correct path for Colab
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)


# 🔍 VECTORSTORE + EMBEDDINGS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever()


# 🤖 LLM + CONTEXTUAL COMPRESSION

llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192")
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor,
)


# 📋 PROMPT TEMPLATE & RAG CHAIN

template = """
You are a helpful assistant that answers questions based on the following context.
If the answer is not in the context, just say you don't know.

Context: {context}
Question: {input}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": compression_retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ⚙️ GRADIO INFERENCE FUNCTION

def answer_question(query):
    try:
        response = rag_chain.invoke(query)
        docs = compression_retriever.get_relevant_documents(query)
        context_used = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return response, context_used
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


# 🎛️ GRADIO INTERFACE

example_questions = [
    "What are points on a mortgage?",
    "Explain fixed-rate vs adjustable-rate mortgage.",
    "What is PMI in mortgage terms?",
    "How does credit score impact mortgage rates?",
    "What is an escrow account?",
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🏠 Mortgage Assistant - RAG LLM")
    gr.Markdown("Ask questions about mortgages based on the document provided.")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(label="Enter your question", placeholder="Ask about mortgage...")
            submit_btn = gr.Button("Submit")
            gr.Examples(example_questions, inputs=question_input)
        with gr.Column():
            response_output = gr.Textbox(label="Answer", lines=5)
            context_output = gr.Textbox(label="Used Context", lines=10, visible=False)
    
    submit_btn.click(fn=answer_question, inputs=question_input, outputs=[response_output, context_output])


# 🚀 LAUNCH

demo.launch()
