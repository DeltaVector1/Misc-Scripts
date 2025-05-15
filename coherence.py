import json
import argparse
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast enough for batch processing

def topic_score(dialog_turns):
    """Score coherence between consecutive turns (0=off-topic, 1=coherent)"""
    if len(dialog_turns) < 2:
        return 1.0  # Single-turn has perfect coherence
    
    embeddings = model.encode([t["value"] for t in dialog_turns])
    scores = []
    for i in range(1, len(embeddings)):
        scores.append(cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0])
    return sum(scores) / len(scores)

def process_file(input_path, output_path, min_coherence=0.5):
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in tqdm(f_in):
            data = json.loads(line)
            if "conversations" not in data:
                continue
            
            score = topic_score(data["conversations"])
            if score >= min_coherence:
                data["topic_coherence"] = score  # Add to JSON
                f_out.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl")
    parser.add_argument("--min-coherence", type=float, default=0.5)
    parser.add_argument("--output", default="filtered.jsonl")
    args = parser.parse_args()
    
    process_file(args.input_jsonl, args.output, args.min_coherence)