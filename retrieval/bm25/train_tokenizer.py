import os
import argparse
from loguru import logger
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

from common import Corpus
from retrieval.datamodule import RetrievalDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--pretrained-tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer to use instead of training (e.g., deepseek-ai/DeepSeek-V3)",
    )
    args = parser.parse_args()
    logger.info(args)

    if args.pretrained_tokenizer:
        # Load a pretrained HuggingFace tokenizer and convert to tokenizers format
        logger.info(f"Loading pretrained tokenizer from {args.pretrained_tokenizer}")
        hf_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)
        
        # The underlying fast tokenizer is already a tokenizers.Tokenizer
        tokenizer = hf_tokenizer.backend_tokenizer
        tokenizer.save(args.output_path)
        logger.info(f"Pretrained tokenizer saved to {args.output_path}")
    else:
        if args.data_path is None:
            raise ValueError("--data-path is required when not using --pretrained-tokenizer")
        
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        )

        corpus = Corpus(os.path.join(args.data_path, "../corpus.jsonl"))
        premises = [premise.serialize() for premise in corpus.all_premises]

        ds_train = RetrievalDataset(
            data_paths=[os.path.join(args.data_path, "train.json")],
            corpus=corpus,
            num_negatives=0,
            num_in_file_negatives=0,
            max_seq_len=1024,
            tokenizer=None,
            is_train=False,
        )
        states = [
            ds_train.data[i]["context"].serialize() for i in range(len(ds_train.data))
        ]
        tokenizer.train_from_iterator(premises + states, trainer=trainer)
        tokenizer.save(args.output_path)
        logger.info(f"Tokenizer saved to {args.output_path}")


if __name__ == "__main__":
    main()
