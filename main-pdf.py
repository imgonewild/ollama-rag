import fitz  # PyMuPDF
# from PyPDF2 import PdfReader
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

# Function to parse text into paragraphs
# def parse_text(text):
#     paragraphs = text.split('\n\n')
#     return [para.strip() for para in paragraphs if para.strip()]

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
    # print(similarity_scores)
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

def main():
    start_time = time.time()

    SYSTEM_PROMPT_TEMPLATE = """
    Answer the question: {context}. 
    ---
    Answer the question based on the above context: {question}. Search the context title first.
    Do not make up answers or use outside information.
    Reply with section titles that are relevant to the answers.
    Reply in the format: {{"answer": "your_answer_here", "source": "your_section_title_here"}} 
    and if the answer contain multiple answers, then combine to one single answer and reply in the format: {{"answer": "answer1, answer2, answer3, etc", "source": "source1, source2, source3, etc"}}
    and if you dont know the answer then reply {{"answer": "I dont know", "source": "N/A"}}
    """

    # Extract text from PDF
    pdf_filename = "ACCU DYNE TES FLUIDS MSDS.pdf"
    # pdf_filename = "Kleiberit 826.0 Cleaner.pdf"
    text = extract_text_from_pdf(pdf_filename)
    paragraphs = parse_text(text)

    # Get embeddings
    embeddings_model = "llama3"
    respond_model = "llama3"
    embeddings = get_embeddings(pdf_filename, embeddings_model , paragraphs)

    # Get user query
    prompt = "What is the signal word?"
    prompt = "What are the first aid measures in case of inhalation?"
    prompt = "What are the first aid measures in case of swallowing?"

    prompt_embedding = ollama.embeddings(model=embeddings_model, prompt=prompt)["embedding"]

    # Find most similar paragraphs
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    context = "\n".join(paragraphs[idx] for _, idx in most_similar_chunks)
    # Format the system prompt with context and question
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, question=prompt)

    # Generate response using ollama
    response = ollama.chat(
        model = respond_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    
    print(pdf_filename)
    print("embeddings_model: ", embeddings_model)
    print("Q: " + prompt)
    print(response["message"]["content"])
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
