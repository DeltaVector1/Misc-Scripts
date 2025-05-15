import json
import argparse
import spacy
import re
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define register features
FORMAL_MARKERS = {
    # Complex/formal conjunctions and transitions
    "words": [
        "accordingly", "consequently", "furthermore", "nevertheless", "therefore",
        "thus", "moreover", "subsequently", "hitherto", "heretofore", "wherein", 
        "thereby", "hereby", "therein", "thereafter", "herewith", "pursuant", 
        "notwithstanding", "aforementioned", "whilst", "amongst", "upon", "whom"
    ],
    # Academic/formal lexical items
    "formal_lexicon": [
        "acquire", "additional", "adequate", "analyze", "assess", "assist", "attain",
        "commence", "conclude", "conduct", "considerable", "demonstrate", "determine",
        "diminish", "distribute", "endeavor", "enhance", "establish", "evaluate", 
        "examine", "facilitate", "fundamental", "implement", "indicate", "initiate",
        "obtain", "participate", "perceive", "perform", "possess", "principal", 
        "procure", "provide", "regarding", "require", "sufficient", "terminate", 
        "utilize", "viable"
    ]
}

INFORMAL_MARKERS = {
    # Contractions
    "contractions": [
        "'ll", "'ve", "'re", "'m", "n't", "'d", "gonna", "wanna", "gotta", "lemme",
        "gimme", "dunno", "kinda", "sorta", "outta", "hafta"
    ],
    # Informal words and slang
    "words": [
        "yeah", "hey", "cool", "okay", "ok", "stuff", "thing", "guys", "like",
        "pretty", "really", "very", "so", "just", "actually", "basically", "literally",
        "totally", "super", "awesome", "amazing", "great", "huge", "massive", "big",
        "kinda", "sorta", "y'know", "anyways", "whatever", "well", "umm", "uh", "hmm"
    ],
    # Informal fillers and hedges
    "fillers": ["um", "uh", "like", "you know", "I mean", "sort of", "kind of"]
}

# Multi-word expression patterns
INFORMAL_PATTERNS = [
    r"\byou guys\b", r"\ba lot\b", r"\bI guess\b", r"\bkind of\b",
    r"\bsort of\b", r"\byou know\b", r"\bI mean\b", r"\bno way\b"
]

FORMAL_PATTERNS = [
    r"\bin accordance with\b", r"\bin addition to\b", r"\bwith regard to\b",
    r"\bin the event that\b", r"\bon the basis of\b", r"\bin respect to\b",
    r"\bin the context of\b", r"\bin conjunction with\b", r"\bby virtue of\b"
]

def analyze_register(text):
    """Analyze linguistic register of a text."""
    doc = nlp(text)
    
    analysis = {
        "formal_features": Counter(),
        "informal_features": Counter(),
        "counts": {
            "formal_markers": 0,
            "informal_markers": 0,
            "sentences": len(list(doc.sents)),
            "words": len([t for t in doc if not t.is_punct and not t.is_space]),
            "first_person_singular": 0,
            "third_person": 0,
            "passive_voice": 0,
            "questions": 0,
            "imperatives": 0,
            "exclamations": 0
        },
        "register_category": "",
        "register_score": 0,  # -1 to 1 scale, -1 very informal, 1 very formal
    }
    
    # Text-level features
    text_lower = text.lower()
    
    # Check multi-word patterns
    for pattern in FORMAL_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            analysis["formal_features"][pattern] += len(matches)
            analysis["counts"]["formal_markers"] += len(matches)
    
    for pattern in INFORMAL_PATTERNS:
        matches = re.findall(pattern, text_lower)
        if matches:
            analysis["informal_features"][pattern] += len(matches)
            analysis["counts"]["informal_markers"] += len(matches)
    
    # Count exclamations and questions
    analysis["counts"]["exclamations"] = text.count("!")
    analysis["counts"]["questions"] = text.count("?")
    
    # Sentence-level analysis
    for sent in doc.sents:
        # Check for imperatives (sentence starts with a verb)
        if len(sent) > 1 and sent[0].pos_ == "VERB":
            analysis["counts"]["imperatives"] += 1
        
        # Check for passive voice
        passive = any(token.dep_ == "auxpass" for token in sent)
        if passive:
            analysis["counts"]["passive_voice"] += 1
    
    # Token-level analysis
    for token in doc:
        # Formal words
        if token.text.lower() in FORMAL_MARKERS["words"] or token.lemma_.lower() in FORMAL_MARKERS["formal_lexicon"]:
            analysis["formal_features"][token.text.lower()] += 1
            analysis["counts"]["formal_markers"] += 1
        
        # Informal words and contractions
        if (token.text.lower() in INFORMAL_MARKERS["words"] or 
            any(contr in token.text.lower() for contr in INFORMAL_MARKERS["contractions"]) or
            token.text.lower() in INFORMAL_MARKERS["fillers"]):
            analysis["informal_features"][token.text.lower()] += 1
            analysis["counts"]["informal_markers"] += 1
        
        # First person singular ("I", "me", "my", "mine")
        if token.text.lower() in ["i", "me", "my", "mine", "myself"]:
            analysis["counts"]["first_person_singular"] += 1
        
        # Third person pronouns and determiners
        if token.text.lower() in ["he", "she", "it", "they", "him", "her", "them", "his", "hers", "its", "their", "theirs"]:
            analysis["counts"]["third_person"] += 1
    
    # Calculate register score (-1 to 1 scale)
    # Normalize by text length to avoid bias
    if analysis["counts"]["words"] > 0:
        formal_ratio = analysis["counts"]["formal_markers"] / analysis["counts"]["words"]
        informal_ratio = analysis["counts"]["informal_markers"] / analysis["counts"]["words"]
        
        # Additional factors to consider in score
        formal_factors = (
            analysis["counts"]["passive_voice"] / max(1, analysis["counts"]["sentences"]) * 0.3 +
            analysis["counts"]["third_person"] / max(1, analysis["counts"]["words"]) * 0.3
        )
        
        informal_factors = (
            analysis["counts"]["first_person_singular"] / max(1, analysis["counts"]["words"]) * 0.2 +
            analysis["counts"]["exclamations"] / max(1, analysis["counts"]["sentences"]) * 0.3 +
            analysis["counts"]["imperatives"] / max(1, analysis["counts"]["sentences"]) * 0.2
        )
        
        # Calculate final score
        analysis["register_score"] = (formal_ratio - informal_ratio + formal_factors - informal_factors) * 5
        
        # Clamp score between -1 and 1
        analysis["register_score"] = max(-1, min(1, analysis["register_score"]))
    
    # Categorize based on score
    if analysis["register_score"] > 0.6:
        analysis["register_category"] = "formal"
    elif analysis["register_score"] > 0.2:
        analysis["register_category"] = "semi-formal"
    elif analysis["register_score"] > -0.2:
        analysis["register_category"] = "neutral"
    elif analysis["register_score"] > -0.6:
        analysis["register_category"] = "casual"
    else:
        analysis["register_category"] = "informal"
    
    # Convert Counter to dict for JSON serialization
    analysis["formal_features"] = dict(analysis["formal_features"])
    analysis["informal_features"] = dict(analysis["informal_features"])
    
    return analysis

def process_line(args):
    """Process a single line from the JSONL file."""
    line, _, _ = args
    obj = json.loads(line)
    
    if "conversations" in obj:
        for turn in obj["conversations"]:
            if turn["from"] in ["human", "gpt"] and turn["value"]:
                turn["register_analysis"] = analyze_register(turn["value"])
                
        # Calculate average register score for the entire conversation
        register_scores = [turn.get("register_analysis", {}).get("register_score", 0) 
                         for turn in obj["conversations"] 
                         if "register_analysis" in turn]
        
        if register_scores:
            obj["avg_register_score"] = sum(register_scores) / len(register_scores)
            
            # Overall conversation register
            if obj["avg_register_score"] > 0.6:
                obj["conversation_register"] = "formal"
            elif obj["avg_register_score"] > 0.2:
                obj["conversation_register"] = "semi-formal"
            elif obj["avg_register_score"] > -0.2:
                obj["conversation_register"] = "neutral"
            elif obj["avg_register_score"] > -0.6:
                obj["conversation_register"] = "casual"
            else:
                obj["conversation_register"] = "informal"
    
    return json.dumps(obj, ensure_ascii=False)

def process_jsonl(input_file, output_file, num_workers, register_filter=None):
    """Process the entire JSONL file."""
    input_path = Path(input_file)
    with input_path.open('r') as f_in, open(output_file, 'w') as f_out:
        total_lines = sum(1 for _ in f_in)
        f_in.seek(0)
        
        with Pool(num_workers) as pool:
            for result in tqdm(pool.imap_unordered(process_line, ((line, False, 0) for line in f_in), chunksize=10), 
                              total=total_lines, desc="Processing"):
                if result:
                    obj = json.loads(result)
                    # Apply register filter if specified
                    if register_filter is None or obj.get("conversation_register") == register_filter:
                        f_out.write(result + '\n')

def analyze_text(text):
    """Analyze a single piece of text."""
    return analyze_register(text)

def main():
    parser = argparse.ArgumentParser(description="Analyze linguistic register in conversations.")
    parser.add_argument('input_file', type=str, help='Input JSONL file path')
    parser.add_argument('-w', '--workers', type=int, default=cpu_count(), help="Number of worker processes")
    parser.add_argument('-r', '--register', type=str, choices=['formal', 'semi-formal', 'neutral', 'casual', 'informal'],
                        help="Filter by register category")
    parser.add_argument('-s', '--single', type=str, help="Analyze a single text instead of a JSONL file")
    
    args = parser.parse_args()
    
    # Single text analysis mode
    if args.single:
        result = analyze_text(args.single)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    
    # JSONL processing mode
    input_path = Path(args.input_file)
    output_path = input_path.parent / f"{input_path.stem}_register_analyzed.jsonl"
    
    print(f"Processing file {input_path}...")
    process_jsonl(input_path, output_path, args.workers, args.register)
    
    # If no filter is applied, generate register distribution report
    if args.register is None:
        print("\nGenerating register distribution report...")
        register_counts = Counter()
        
        with open(output_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                if "conversation_register" in obj:
                    register_counts[obj["conversation_register"]] += 1
        
        total = sum(register_counts.values())
        if total > 0:
            print("\nRegister Category\tCount\tPercentage")
            print("----------------------------------------")
            for category in ['formal', 'semi-formal', 'neutral', 'casual', 'informal']:
                count = register_counts[category]
                percentage = (count / total) * 100
                print(f"{category}\t\t{count}\t{percentage:.1f}%")

if __name__ == "__main__":
    main()