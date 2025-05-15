    import json
import argparse
import re
from pathlib import Path
from textstat import flesch_kincaid_grade, smog_index, coleman_liau_index

def calculate_readability(text):
    return {
        "flesch_kincaid": flesch_kincaid_grade(text),
        "smog": smog_index(text),
        "coleman_liau": coleman_liau_index(text)
    }

def process_file(input_path, output_path, max_grade=8):
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            data = json.loads(line)
            if "conversations" not in data:
                continue
            
            keep = True
            for turn in data["conversations"]:
                if turn["from"] in ["human", "gpt"]:
                    analysis = calculate_readability(turn["value"])
                    # Reject if any score exceeds threshold
                    if any(score > max_grade for score in analysis.values()):
                        keep = False
                        break
                    turn["readability"] = analysis
            
            if keep:
                f_out.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--max-grade", type=float, default=8.0, 
                       help="Maximum allowed reading grade level (default: 8)")
    parser.add_argument("--output", default="filtered.jsonl")
    args = parser.parse_args()
    
    process_file(args.input, args.output, args.max_grade)