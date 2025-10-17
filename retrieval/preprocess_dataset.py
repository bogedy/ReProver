"""Script to prebuild and serialize datasets to disk."""

import os
import json
import pickle
import argparse

from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from common import Corpus, IndexedCorpus
from retrieval.datamodule import RetrievalDataset


def build_fast_accessible_cache(corpus, unique_contexts):
    """Build cache efficiently by precomputing per-path premise indexes."""
    logger.info("Precomputing per-path premise indexes...")
    
    # Build a mapping: path -> list of (premise_index, end_pos)
    path_to_premises = {}
    for i, p in enumerate(tqdm(corpus.all_premises, desc="Indexing premises by path")):
        if p.path not in path_to_premises:
            path_to_premises[p.path] = []
        path_to_premises[p.path].append((i, p.end))
    
    # Build a mapping: path -> set of premise indexes from imported files
    logger.info("Computing imported premise indexes per path...")
    path_to_imported_indexes = {}
    for path in tqdm(corpus.transitive_dep_graph.nodes, desc="Computing imports"):
        imported_indexes = []
        for dep_path in corpus.transitive_dep_graph.successors(path):
            if dep_path in path_to_premises:
                imported_indexes.extend([idx for idx, _ in path_to_premises[dep_path]])
        path_to_imported_indexes[path] = set(imported_indexes)
    
    # Now build the cache for each unique context
    logger.info(f"Building cache for {len(unique_contexts)} contexts...")
    accessible_premise_indexes_cache = {}
    
    for path, pos in tqdm(unique_contexts, desc="Building accessible premise cache"):
        # Get premises from same file that end before pos
        same_file_indexes = [
            idx for idx, end_pos in path_to_premises.get(path, [])
            if end_pos <= pos
        ]
        
        # Get all imported premises
        imported_indexes = path_to_imported_indexes.get(path, set())
        
        # Combine both
        accessible_indexes = same_file_indexes + list(imported_indexes)
        accessible_premise_indexes_cache[(path, pos)] = accessible_indexes
    
    return accessible_premise_indexes_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="kaiyuy/leandojo-lean4-retriever-byt5-small")
    parser.add_argument("--num_negatives", type=int, default=3)
    parser.add_argument("--num_in_file_negatives", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"])
    parser.add_argument("--indexed_corpus_path", type=str, required=False, help="Path to a pickled indexed corpus.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    corpus = Corpus(args.corpus_path)

    # for caching accessible premises
    unique_contexts = set()
    
    for split in args.splits:
        logger.info(f"Processing {split} split...")
        
        is_train = (split == "train")
        dataset = RetrievalDataset(
            data_paths=[os.path.join(args.data_path, f"{split}.json")],
            corpus=corpus,
            num_negatives=args.num_negatives,
            num_in_file_negatives=args.num_in_file_negatives,
            max_seq_len=args.max_seq_len,
            tokenizer=tokenizer,
            is_train=is_train,
        )

        for ex in dataset.data:
            ctx = ex["context"]
            unique_contexts.add((ctx.path, ctx.theorem_pos))
        
        output_path = os.path.join(args.output_dir, f"{split}_dataset.pkl")
        logger.info(f"Saving {len(dataset)} examples to {output_path}")
        with open(output_path, "wb") as f:
            pickle.dump(dataset.data, f)
        
        logger.info(f"✓ Saved {split} dataset")

    # Build the cache efficiently
    accessible_premise_indexes_cache = build_fast_accessible_cache(corpus, unique_contexts)
    
    corpus.accessible_premise_indexes_cache = accessible_premise_indexes_cache
    logger.info(f"Built cache with {len(accessible_premise_indexes_cache)} entries")

    # Save the corpus once
    corpus_path = os.path.join(args.output_dir, "corpus.pkl")
    logger.info(f"Saving corpus to {corpus_path}")
    with open(corpus_path, "wb") as f:
        pickle.dump(corpus, f)

    if args.indexed_corpus_path is not None:
        logger.info(f"Loading indexed corpus from {args.indexed_corpus_path}")
        with open(args.indexed_corpus_path, "rb") as f:
            indexed_corpus = pickle.load(f)
        indexed_corpus = IndexedCorpus(corpus=corpus, embeddings=indexed_corpus.embeddings)
        with open(args.indexed_corpus_path, "wb") as f:
            pickle.dump(indexed_corpus, f)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
