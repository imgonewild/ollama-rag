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
    Reply in the format: {{"answer": "your_answer_here", "source": "your_section_title_here"}} 
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
    Check the answer for correctness, completeness, and relevance to the context.
    Reply in the format: {{"answer": "your_answer_here", "source": "your_section_title_here"}} 
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
    Here is the refined answer.
    Ensure that the refined answer addresses any issues pointed out in the critique, and reply in the format: {{"answer": "your_answer_here", "source": "your_section_title_here"}} 
    """
    refined_response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": improvement_prompt},
        ],
    )
    refined_answer = refined_response["message"]["content"]

    # Check if the refined answer is the same as the initial answer
    if refined_answer == initial_answer:
        return f'No changes were needed. Initial and refined answers are the same: {initial_answer}'
    
    return refined_answer
def main():
    start_time = time.time()

    # Extract text from PDF
    pdf_filename = "Adhesive, 3M Spray Adhesive 90 SDS (aerosol).pdf"
    text = extract_text_from_pdf(pdf_filename)
    paragraphs = parse_text(text)

    # Get embeddings
    embeddings_model = "llama3"
    respond_model = "llama3"
    embeddings = get_embeddings(pdf_filename, embeddings_model , paragraphs)

    # Get user query
    prompt = "What are the first aid measures in case of swallowing?"
    prompt = "What are the first aid measures in case of inhalation?"
    prompt = "What is the signal word?"

    prompt_embedding = ollama.embeddings(model=embeddings_model, prompt=prompt)["embedding"]

    # Find most similar paragraphs
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]
    context = "\n".join(paragraphs[idx] for _, idx in most_similar_chunks)
    f = open(pdf_filename+".txt", "a")
    f.write(context)
    f.close()

    # Generate initial answer
    initial_answer = generate_initial_answer(context, prompt, respond_model)
    
    # Refine the answer using reflection
    refined_answer = refine_answer(context, initial_answer, respond_model)
    # refined_answer_2 = refine_answer(context, refined_answer, respond_model)

    print(pdf_filename)
    print("embeddings_model: ", embeddings_model)
    print("Q: " + prompt)
    print("Initial Answer: " + initial_answer)
    print("Refined Answer: " + refined_answer)
    # print("2nd Refined Answer: " + refined_answer_2)

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()