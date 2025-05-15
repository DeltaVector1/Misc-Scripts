import os
import requests
import json
import re
import time
import nltk
# from nltk.tokenize import sent_tokenize # Not using sent_tokenize directly here

# --- Configuration ---
PDF_URL = "https://arxiv.org/pdf/2307.09288.pdf" # Example PDF URL, change it
OUTPUT_DIR = "output_data_multi_turn"
PDF_FILENAME = os.path.join(OUTPUT_DIR, "downloaded.pdf")
TEXT_FILENAME = os.path.join(OUTPUT_DIR, "parsed_text.txt")
FINAL_JSON_FILENAME = os.path.join(OUTPUT_DIR, "final_sharegpt_multi_turn.json")

# vLLM Server endpoint (Use the CHAT endpoint logic from previous step)
VLLM_ENDPOINT_CHAT = "http://localhost:8000/v1/chat/completions"

# ** MODIFIED ** Generation Prompt for Multi-Turn
# We need a very specific format for the LLM to follow.
MULTI_TURN_GEN_PROMPT_TEMPLATE = """Based *only* on the provided Text Snippet, generate a short, relevant conversation with exactly two turns (Human asks, AI answers, Human asks follow-up, AI answers). Do not add any preamble, explanation, or text other than the conversation itself. Use the following format strictly, including the role tags:
Human: [First Question related to the text]
AI: [Answer based *only* on the text]
Human: [Follow-up question building on the first answer, answerable from the text]
AI: [Answer to follow-up based *only* on the text]

Text Snippet:
{chunk_text}"""

# ** MODIFIED ** Curation Prompt for Multi-Turn Conversation
CURATION_PROMPT_TEMPLATE = """On a scale of 1 to 10, how relevant, coherent, and accurate is the following multi-turn Conversation based *solely* on the provided Context Text? Consider if the questions are answerable from the text and if the answers are factually correct according to the text. Output *only* the numeric rating (e.g., "8").

Context Text:
{chunk_text}

Conversation:
{conversation_text}

Rating (1-10):"""

CURATION_THRESHOLD = 6.0 # Maybe lower threshold for multi-turn? Adjust as needed.
MAX_TOKENS_GEN = 500 # ** INCREASED ** Greatly increase max tokens for longer conversation
MAX_TOKENS_RATE = 10
TEMPERATURE = 0.75 # Slightly higher temp might help conversation flow?
TOP_P = 0.95

# Chunk Cfg
MIN_CHUNK_WORDS = 60 # Slightly larger chunks might be better for multi-turn context

# --- PDF Parsing (Same as before) ---
try:
    from pdfminer.high_level import extract_text
except ImportError:
    print("Error: pdfminer.six is required. Install it with: pip install pdfminer.six")
    exit(1)

class PDFParser:
    def parse(self, file_path: str) -> str:
        print(f"Parsing PDF: {file_path}")
        try:
            text = extract_text(file_path)
            print(f"Successfully extracted text.")
            # Basic cleaning: replace multiple newlines/spaces more aggressively
            text = re.sub(r'(\s*\n){3,}', '\n\n', text) # Reduce 3+ newlines to 2
            text = re.sub(r'[ \t]+', ' ', text) # Reduce multiple spaces/tabs to one
            text = re.sub(r'\n ', '\n', text) # Remove space after newline
            return text
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            raise

    def save(self, content: str, output_path: str) -> None:
        print(f"Saving parsed text to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("Text saved.")

# --- Text Chunking (Same as before) ---
def chunk_text_by_paragraphs(text: str, min_words: int = 60) -> list[str]:
    print("Chunking text into paragraphs...")
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    valid_chunks = [p for p in paragraphs if len(p.split()) >= min_words]
    print(f"Found {len(paragraphs)} paragraphs, {len(valid_chunks)} are >= {min_words} words.")
    return valid_chunks

# --- vLLM Interaction (Using Chat Endpoint logic) ---
def query_vllm(prompt: str, max_tokens: int, temperature: float = 0.1, top_p: float = 1.0) -> str:
    # ALWAYS use chat endpoint format now
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": "unsloth/Mistral-Small-Instruct-2409-bnb-4bit",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": ["\n\n\n"] # Maybe add a stop sequence if model tends to ramble? Adjust as needed.
    }
    endpoint = VLLM_ENDPOINT_CHAT
    headers = {"Content-Type": "application/json"}
    try:
        # print(f"Querying vLLM endpoint: {endpoint} with prompt:\n{prompt[:100]}...") # DEBUG
        response = requests.post(endpoint, json=payload, headers=headers, timeout=180) # Increased timeout
        response.raise_for_status()
        result = response.json()

        if 'choices' in result and len(result['choices']) > 0 and 'message' in result['choices'][0] and 'content' in result['choices'][0]['message']:
             return result['choices'][0]['message']['content'].strip()
        else:
             print(f"Warning: Unexpected vLLM chat response format: {result}")
             return ""
    except requests.exceptions.RequestException as e:
        print(f"Error querying vLLM ({endpoint}): {e}")
        # Add more detail for debugging
        if hasattr(e, 'response') and e.response is not None:
             print(f"Response status: {e.response.status_code}")
             print(f"Response text: {e.response.text}")
        return ""

# --- ** NEW ** Conversation Parsing ---
def parse_conversation_response(response_text: str) -> list[dict] | None:
    """
    Parses a multi-turn conversation assuming the format:
    Human: Text...
    AI: Text...
    Human: Text...
    AI: Text...
    """
    print(f"Attempting to parse conversation:\n{response_text[:200]}...") # Debug
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]
    conversation = []
    expected_roles = ["Human", "AI"]
    role_index = 0

    current_role = None
    current_content = []

    for line in lines:
        found_role = False
        for role_tag in expected_roles:
            if line.lower().startswith(role_tag.lower() + ":"):
                # If we were building content for a previous role, save it
                if current_role and current_content:
                    conversation.append({"role": current_role.lower(), "content": "\n".join(current_content).strip()})
                    current_content = []

                # Start the new role
                current_role = role_tag # Keep original casing for the key
                content_start_index = len(role_tag) + 1 # Index after "Role:"
                current_content.append(line[content_start_index:].strip())
                found_role = True
                break # Found the role for this line

        if not found_role and current_role:
             # This line is a continuation of the previous role's content
             current_content.append(line)

    # Append the last collected role and content
    if current_role and current_content:
         conversation.append({"role": current_role.lower(), "content": "\n".join(current_content).strip()})


    # Basic validation: Check if we have at least one turn for each expected role
    roles_present = {turn['role'] for turn in conversation}
    if 'human' in roles_present and ('ai' in roles_present or 'gpt' in roles_present ): # Accept 'ai' or 'gpt' for model response
        print(f"  -> Successfully parsed {len(conversation)} turns.")
        # Map 'ai' role to 'gpt' for ShareGPT standard if needed
        for turn in conversation:
            if turn['role'] == 'ai':
                turn['role'] = 'gpt'
        return conversation
    else:
        print(f"Warning: Could not parse a valid Human/AI conversation structure. Found roles: {roles_present}")
        return None


# --- Rating Parsing (Same as before) ---
def parse_rating_response(response_text: str) -> float | None:
    match = re.search(r"(\d+(\.\d+)?)", response_text)
    if match:
        try:
            rating = float(match.group(1))
            if 1.0 <= rating <= 10.0:
                return rating
            else:
                print(f"Warning: Parsed rating {rating} out of range (1-10).")
        except ValueError:
            print(f"Warning: Could not convert parsed rating '{match.group(1)}' to float.")
    print(f"Warning: Could not parse numeric rating from response: '{response_text}'")
    return None

# --- Main Logic (Modified for Multi-Turn) ---
def main():
    # 1. Download PDF (Same)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading PDF from {PDF_URL}...")
    try:
        response = requests.get(PDF_URL, stream=True)
        response.raise_for_status()
        with open(PDF_FILENAME, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"PDF saved to {PDF_FILENAME}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download PDF: {e}")
        exit(1)

    # 2. Parse PDF to Text (Same)
    parser = PDFParser()
    try:
        pdf_text = parser.parse(PDF_FILENAME)
        parser.save(pdf_text, TEXT_FILENAME)
    except Exception as e:
        print(f"Failed during PDF parsing or saving: {e}")
        exit(1)

    # 3. Chunk Text (Same)
    text_chunks = chunk_text_by_paragraphs(pdf_text, min_words=MIN_CHUNK_WORDS)
    if not text_chunks:
          print("No suitable text chunks found after parsing. Exiting.")
          exit(1)

    # 4. Generate Conversations (Modified)
    print("\n--- Generating Multi-Turn Conversations ---")
    generated_conversations = []
    for i, chunk in enumerate(text_chunks):
        print(f"Processing chunk {i+1}/{len(text_chunks)} (Length: {len(chunk)} chars)...")
        prompt = MULTI_TURN_GEN_PROMPT_TEMPLATE.format(chunk_text=chunk)
        response = query_vllm(prompt, max_tokens=MAX_TOKENS_GEN, temperature=TEMPERATURE, top_p=TOP_P)
        if response:
            # Use the new parser
            parsed_convo = parse_conversation_response(response)
            if parsed_convo:
                generated_conversations.append({
                    "chunk": chunk,
                    "conversation": parsed_convo, # Store the list of turns
                    "raw_response": response # Keep raw for curation prompt
                })
                print(f"  -> Generated conversation.")
            else:
                 print(f"  -> Failed to parse conversation structure for chunk {i+1}.")
                 # print(f"Failed Response:\n{response}\n-----------------") # Uncomment for debugging failed parses
        else:
            print(f"  -> Failed to get response for chunk {i+1}.")
        time.sleep(1.0) # Maybe increase sleep time for longer generation

    print(f"\nGenerated {len(generated_conversations)} raw conversations.")

    # 5. Curate Conversations (Modified)
    print("\n--- Curating Conversations ---")
    curated_conversations = []
    for i, item in enumerate(generated_conversations):
        print(f"Curating conversation {i+1}/{len(generated_conversations)}...")
        # Use the raw response text for the curation prompt as it was generated
        conversation_text_for_rating = item['raw_response']
        prompt = CURATION_PROMPT_TEMPLATE.format(
            chunk_text=item["chunk"],
            conversation_text=conversation_text_for_rating
        )
        response = query_vllm(prompt, max_tokens=MAX_TOKENS_RATE, temperature=0.1) # Low temp for rating
        if response:
            rating = parse_rating_response(response)
            if rating is not None:
                print(f"  -> Rating: {rating}")
                if rating >= CURATION_THRESHOLD:
                    # Store the parsed conversation turns, not the raw response
                    curated_conversations.append(item['conversation'])
                    print(f"  -> KEPT (>= {CURATION_THRESHOLD})")
                else:
                    print(f"  -> DISCARDED (< {CURATION_THRESHOLD})")
            else:
                 print(f"  -> Failed to parse rating for conversation {i+1}.")
                 print(f"Rating response: {response}") # Debug
        else:
            print(f"  -> Failed to get rating response for conversation {i+1}.")
        time.sleep(0.5)

    print(f"\nKept {len(curated_conversations)} curated conversations.")

    # 6. Save as ShareGPT Format (Modified)
    print("\n--- Saving Final Data ---")
    sharegpt_data = []
    for conversation_turns in curated_conversations:
        # Ensure turns alternate correctly if needed, parser maps 'ai' to 'gpt'
        formatted_turns = [{"from": turn['role'], "value": turn['content']} for turn in conversation_turns]
        sharegpt_data.append({"conversations": formatted_turns})

    if sharegpt_data:
        try:
            with open(FINAL_JSON_FILENAME, 'w', encoding='utf-8') as f:
                json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {len(sharegpt_data)} conversations to {FINAL_JSON_FILENAME}")
        except IOError as e:
            print(f"Error saving final JSON file: {e}")
    else:
        print("No curated conversations to save.")

    print("\nScript finished. Go bother someone else.")

if __name__ == "__main__":
    # NLTK Check (same as before)
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("NLTK 'punkt' tokenizer not found. Please download it.")
        print("Run python and execute: import nltk; nltk.download('punkt')")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred checking NLTK data: {e}")
        exit(1)

    main()