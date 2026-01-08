import argparse
import json
import random
import numpy as np
import sys
import os

# Add the parent directory (ReProver/) to sys.path to allow importing common.py
# This assumes the script is located in ReProver/retrieval/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from transformers import AutoTokenizer
    from tqdm import tqdm
    from lean_dojo import Pos
    from common import Premise
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure you are running this script in an environment with the necessary dependencies installed.")
    print("And that common.py is accessible (typically in the parent directory).")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths of premises in the corpus.")
    parser.add_argument("--corpus-path", type=str, required=True, help="Path to the corpus.jsonl file.")
    parser.add_argument("--model-name", type=str, default="intfloat/e5-small-v2", help="Name of the model to use for tokenization.")
    parser.add_argument("--sample-files", type=int, default=1000, help="Number of files (lines) to randomly sample from the corpus.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading tokenizer for model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Reading corpus from: {args.corpus_path}")
    try:
        with open(args.corpus_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Corpus file not found at {args.corpus_path}")
        return

    total_files = len(lines)
    print(f"Total files in corpus: {total_files}")
    
    sample_size = min(total_files, args.sample_files)
    print(f"Sampling {sample_size} files...")
    sampled_lines = random.sample(lines, sample_size)

    premises = []
    print("Extracting premises...")
    for line in sampled_lines:
        try:
            file_data = json.loads(line)
            path = file_data.get("path", "")
            
            for p in file_data.get("premises", []):
                full_name = p.get("full_name")
                code = p.get("code")
                start = p.get("start")
                end = p.get("end")

                # Filter out invalid premises (logic from common.File.from_data)
                if full_name is None:
                    continue
                if "user__.n" in full_name or code == "":
                    continue
                if full_name.startswith("[") and full_name.endswith("]"):
                    continue
                
                # Construct Premise object to use its serialize method
                # This ensures we count tokens exactly as the model sees them
                premise = Premise(
                    path=path,
                    full_name=full_name,
                    start=Pos(*start),
                    end=Pos(*end),
                    code=code
                )
                premises.append(premise)
        except json.JSONDecodeError:
            print("Warning: Skipping malformed JSON line.")
            continue
        except Exception as e:
            # Catch other potential errors (e.g. malformed start/end)
            continue

    if not premises:
        print("No valid premises found in the sampled files.")
        return

    print(f"Extracted {len(premises)} valid premises. Tokenizing...")

    token_counts = []
    exceeds_512 = 0
    exceeds_1024 = 0
    exceeds_2048 = 0

    for premise in tqdm(premises):
        # serialize() adds special tokens/tags that the retriever uses
        text = premise.serialize()
        tokens = tokenizer.encode(text, add_special_tokens=True)
        length = len(tokens)
        token_counts.append(length)
        
        if length > 512:
            exceeds_512 += 1
        if length > 1024:
            exceeds_1024 += 1
        if length > 2048:
            exceeds_2048 += 1

    token_counts = np.array(token_counts)

    print("\n" + "="*30)
    print(f"Token Length Analysis for {args.model_name}")
    print("="*30)
    print(f"Files Sampled:      {sample_size}")
    print(f"Premises Analyzed:  {len(premises)}")
    print(f"Mean Length:        {np.mean(token_counts):.2f}")
    print(f"Median Length:      {np.median(token_counts):.2f}")
    print(f"Min Length:         {np.min(token_counts)}")
    print(f"Max Length:         {np.max(token_counts)}")
    print("-" * 30)
    print(f"Percentiles:")
    print(f"  90th: {np.percentile(token_counts, 90):.2f}")
    print(f"  95th: {np.percentile(token_counts, 95):.2f}")
    print(f"  99th: {np.percentile(token_counts, 99):.2f}")
    print("-" * 30)
    print(f"Count > 512:        {exceeds_512} ({exceeds_512/len(premises)*100:.2f}%)")
    print(f"Count > 1024:       {exceeds_1024} ({exceeds_1024/len(premises)*100:.2f}%)")
    print(f"Count > 2048:       {exceeds_2048} ({exceeds_2048/len(premises)*100:.2f}%)")
    print("="*30)

if __name__ == "__main__":
    main()
