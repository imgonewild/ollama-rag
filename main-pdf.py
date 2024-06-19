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
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

# Function to load embeddings
def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
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

    SYSTEM_PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}. Do not make up answers or use outside information.

    ---

    Answer the question based on the above context: {question}. Do not make up answers or use outside information. Reply with section titles that are relevant to the answers.
    Reply in the format: {{"answer": "your_answer_here", "source": "your_section_title_here"}}
    """

    # Extract text from PDF
    pdf_filename = "Adhesive, 3M Spray Adhesive 90 SDS (aerosol).pdf"
    text = extract_text_from_pdf(pdf_filename)
    paragraphs = parse_text(text)

    # Get embeddings
    embeddings = get_embeddings(pdf_filename, "llama3", paragraphs)

    # Get user query
    prompt = "What is the product description?"
    
    prompt_embedding = ollama.embeddings(model="llama3", prompt=prompt)["embedding"]

    # Find most similar paragraphs
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    context = "\n".join(paragraphs[idx] for _, idx in most_similar_chunks)

    # Format the system prompt with context and question
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, question=prompt)

    # Generate response using ollama
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    print(response["message"]["content"])
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
