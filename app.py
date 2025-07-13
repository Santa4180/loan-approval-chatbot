import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from transformers import pipeline


df = pd.read_csv("Training Dataset.csv").dropna()


docs = [
    f"Applicant {i}: Gender={row['Gender']}, Married={row['Married']}, Education={row['Education']}, Self_Employed={row['Self_Employed']}, Credit_History={row['Credit_History']}, Property_Area={row['Property_Area']}, Loan_Status={row['Loan_Status']}"
    for i, row in df.iterrows()
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, convert_to_tensor=False)

index = faiss.IndexFlatL2(doc_embeddings[0].shape[0])
index.add(np.array(doc_embeddings))

def retrieve(query, k=5):
    query_vec = embedder.encode([query])[0]
    D, I = index.search(np.array([query_vec]), k)
    return [docs[i] for i in I[0]]

generator = pipeline("text-generation", model="tiiuae/falcon-rw-1b", max_new_tokens=150)

st.title("Loan Approval Chatbot")
query = st.text_input("Ask a question related to loan approval dataset:")

if query:
    relevant_docs = retrieve(query)
    context = "\n".join(relevant_docs)
    prompt = f"Context:\n{context}\n\nQ: {query}\nA:"
    result = generator(prompt)[0]['generated_text']
    st.write(result.split("A:")[-1].strip())
