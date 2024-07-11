import fitz  # PyMuPDF
import os
import json
import time
import ollama
import numpy as np
from numpy.linalg import norm
from langchain.text_splitter import CharacterTextSplitter 

# Function to extract text from PDF
def extract_text_from_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to parse text into chunks
def parse_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=250,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

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

# Function to generate the initial answer
def generate_initial_answer(context, question, model):
    system_prompt = f"""
    Answer the question: {context}. 
    ---
    Answer the question based on the above context: {question}. 
    Do not make up answers or use outside information.
    Reply in the format: {{"answer": "your_answer_here"}}
    """
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
    )
    return response["message"]["content"]

# Function to critique and refine the answer
def refine_answer(context, initial_answer, model):
    critique_prompt = f"""
    Here is the initial answer: {initial_answer}
    Context: {context}
    Check the answer for correctness, completeness, and provide constructive criticism.
    """
    critique_response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": critique_prompt},
        ],
    )
    critique = critique_response["message"]["content"]

    improvement_prompt = f"""
    Using the following critique, refine the initial answer: {critique}
    Context: {context}
    Here is the refined answer: 
    """
    refined_response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": improvement_prompt},
        ],
    )
    return refined_response["message"]["content"]

def main():
    start_time = time.time()

    # Extract text from PDF
    pdf_filename = "ACCU DYNE TES FLUIDS MSDS.pdf"
    text = extract_text_from_pdf(pdf_filename)
    paragraphs = parse_text(text)

    # Get embeddings
    embeddings_model = "llama3"
    respond_model = "llama3"
    embeddings = get_embeddings(pdf_filename, embeddings_model , paragraphs)

    # Get user query
    prompt = "What are the first aid measures in case of swallowing?"

    prompt_embedding = ollama.embeddings(model=embeddings_model, prompt=prompt)["embedding"]

    # Find most similar paragraphs
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    context = "\n".join(paragraphs[idx] for _, idx in most_similar_chunks)

    # Generate initial answer
    initial_answer = generate_initial_answer(context, prompt, respond_model)
    
    # Refine the answer using reflection
    refined_answer = refine_answer(context, initial_answer, respond_model)

    print(pdf_filename)
    print("embeddings_model: ", embeddings_model)
    print("Q: " + prompt)
    print("Initial Answer: " + initial_answer)
    print("Refined Answer: " + refined_answer)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()