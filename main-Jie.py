import os 
import json
import time
import ollama


def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs = []
        buffer = []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                # paragraphs.append((" ").join(buffer))
                paragraphs.append(" ".join(buffer))
                buffer = []
        if len(buffer):
            # aragraphs.append((" ")).join((buffer))
            paragraphs.append(" ".join(buffer))
        return paragraphs

def save_embeddings(filename, embeddings):
    #create dir if it doesn't exist
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    #dump embeddings to json
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)

def get_embeddings(filename, modelname, chunks):
    #check if embeddings are already saved
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    #get embeddings from ollama
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk) ["embedding"]
        for chunk in chunks
    ]
    #save embeddings
    save_embeddings(filename, embeddings)
    return embeddings

def load_embeddings(filename):
    #check if file exists
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    #load embeddings from json
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)
    
def main():
    #open file
    filename = "peter-pan.txt"
    paragraphs = parse_file(filename)
    start = time.perf_counter()
    embeddings = get_embeddings(filename, "llama3", paragraphs)
    print(time.perf_counter() - start)
    print(len(embeddings))


if __name__ == "__main__":
    main()