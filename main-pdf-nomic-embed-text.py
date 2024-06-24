import fitz  # PyMuPDF
import os
import json
import time
import ollama
import numpy as np
from numpy.linalg import norm

# Function to extract text from PDF
def extract_text_from_pdf(filename):
    doc = fitz.open(filename)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to parse text into paragraphs
def parse_text(text):
    paragraphs = text.split('\n\n')
    return [para.strip() for para in paragraphs if para.strip()]

# Function to save embeddings
def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings-nomic"):
        os.makedirs("embeddings-nomic")
    with open(f"embeddings-nomic/{filename}.json", "w") as f:
        json.dump(embeddings, f)

# Function to load embeddings
def load_embeddings(filename):
    if not os.path.exists(f"embeddings-nomic/{filename}.json"):
        return False
    with open(f"embeddings-nomic/{filename}.json", "r") as f:
        return json.load(f)

# Function to get embeddings using ollama
def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

# Function to find most similar embeddings
def find_most_similar(needle, haystack):
    needle_norm = norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    start_time = time.time()
    SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    
    # Extract text from PDF
    pdf_filename = "Kleiberit 826.0 Cleaner.pdf"
    text = extract_text_from_pdf(pdf_filename)
    paragraphs = parse_text(text)

    # Get embeddings
    embeddings = get_embeddings(pdf_filename, "nomic-embed-text", paragraphs)

    # Get user query
    prompt = "What are the hazard classifications associated with the product?"
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]

    # Find most similar paragraphs
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    context = "\n".join(paragraphs[idx] for _, idx in most_similar_chunks)

    # Generate response using ollama
    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print(response["message"]["content"])
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
