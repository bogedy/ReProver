"""Datamodule for the premise retrieval."""

import os
import json
import torch
import pickle
import random
import itertools
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from lean_dojo import Pos
import pytorch_lightning as pl
from lean_dojo import LeanGitRepo
from typing import Optional, List
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader


from common import Context, Corpus, Batch, Example, get_all_pos_premises, IndexedCorpus


class RetrievalDataset(Dataset):
    def __init__(
        self,
        data_paths: List[str],
        corpus: Corpus,
        num_negatives: int,
        num_in_file_negatives: int,
        max_seq_len: int,
        tokenizer,
        is_train: bool,
    ) -> None:
        super().__init__()
        self.corpus = corpus
        self.num_negatives = num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.data = list(
            itertools.chain.from_iterable(self._load_data(path) for path in data_paths)
        )

    def _load_data(self, data_path: str) -> List[Example]:
        data = []
        logger.info(f"Loading data from {data_path}")

        for thm in tqdm(json.load(open(data_path))):
            file_path = thm["file_path"]

            for i, tac in enumerate(thm["traced_tactics"]):
                context = Context(
                    file_path, thm["full_name"], Pos(*thm["start"]), tac["state_before"]
                )
                all_pos_premises = get_all_pos_premises(
                    tac["annotated_tactic"], self.corpus
                )

                if self.is_train:
                    # In training, we ignore tactics that do not have any premises.
                    for pos_premise in all_pos_premises:
                        data.append(
                            {
                                "url": thm["url"],
                                "commit": thm["commit"],
                                "file_path": thm["file_path"],
                                "full_name": thm["full_name"],
                                "start": thm["start"],
                                "tactic_idx": i,
                                "context": context,
                                "pos_premise": pos_premise,
                                "all_pos_premises": all_pos_premises,
                            }
                        )
                else:
                    data.append(
                        {
                            "url": thm["url"],
                            "commit": thm["commit"],
                            "file_path": thm["file_path"],
                            "full_name": thm["full_name"],
                            "start": thm["start"],
                            "tactic_idx": i,
                            "context": context,
                            "all_pos_premises": all_pos_premises,
                        }
                    )

        logger.info(f"Loaded {len(data)} examples.")
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Example:
        if not self.is_train:
            return self.data[idx]

        # In-file negatives + random negatives from all accessible premises.
        ex = deepcopy(self.data[idx])
        premises_in_file = []
        premises_outside_file = []

        for p in self.corpus.get_premises(ex["context"].path):
            if p == ex["pos_premise"]:
                continue
            if p.end < ex["context"].theorem_pos:
                if ex["pos_premise"].path == ex["context"].path:
                    premises_in_file.append(p)
                else:
                    premises_outside_file.append(p)

        for p in self.corpus.transitive_dep_graph.successors(ex["context"].path):
            if p == ex["pos_premise"].path:
                premises_in_file += [
                    _p for _p in self.corpus.get_premises(p) if _p != ex["pos_premise"]
                ]
            else:
                premises_outside_file += self.corpus.get_premises(p)

        num_in_file_negatives = min(len(premises_in_file), self.num_in_file_negatives)

        ex["neg_premises"] = random.sample(
            premises_in_file, num_in_file_negatives
        ) + random.sample(
            premises_outside_file, self.num_negatives - num_in_file_negatives
        )
        return ex

    def collate(self, examples: List[Example]) -> Batch:
        batch = {}

        # Tokenize the context.
        context = [ex["context"] for ex in examples]
        tokenized_context = self.tokenizer(
            [c.serialize() for c in context],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        batch["context"] = context
        batch["context_ids"] = tokenized_context.input_ids
        batch["context_mask"] = tokenized_context.attention_mask

        # Tokenize the label and premises.
        if self.is_train:
            pos_premise = [ex["pos_premise"] for ex in examples]
            tokenized_pos_premise = self.tokenizer(
                [p.serialize() for p in pos_premise],
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            batch["pos_premise"] = pos_premise
            batch["pos_premise_ids"] = tokenized_pos_premise.input_ids
            batch["pos_premise_mask"] = tokenized_pos_premise.attention_mask

            batch_size = len(examples)
            label = torch.zeros(batch_size, batch_size * (1 + self.num_negatives))

            # Check if one's negative is another's positive
            for j in range(batch_size):
                all_pos_premises = examples[j]["all_pos_premises"]
                for k in range(batch_size * (1 + self.num_negatives)):
                    if k < batch_size:
                        pos_premise_k = examples[k]["pos_premise"]
                    else:
                        pos_premise_k = examples[k % batch_size]["neg_premises"][
                            k // batch_size - 1
                        ]
                    label[j, k] = float(pos_premise_k in all_pos_premises)

            batch["label"] = label
            batch["neg_premises"] = []
            batch["neg_premises_ids"] = []
            batch["neg_premises_mask"] = []

            for i in range(self.num_negatives):
                neg_premise = [ex["neg_premises"][i] for ex in examples]
                tokenized_neg_premise = self.tokenizer(
                    [p.serialize() for p in neg_premise],
                    padding="longest",
                    max_length=self.max_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                batch["neg_premises"].append(neg_premise)
                batch["neg_premises_ids"].append(tokenized_neg_premise.input_ids)
                batch["neg_premises_mask"].append(tokenized_neg_premise.attention_mask)

        # Copy the rest of the fields.
        for k in examples[0].keys():
            if k not in batch:
                batch[k] = [ex[k] for ex in examples]

        return batch


class RetrievalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        corpus_path: str,
        num_negatives: int,
        num_in_file_negatives: int,
        model_name: str,
        batch_size: int,
        eval_batch_size: int,
        max_seq_len: int,
        num_workers: int,
        prebuilt_data_path: Optional[str] = None,
        prebuilt_corpus_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.num_negatives = num_negatives
        assert 0 <= num_in_file_negatives <= num_negatives
        self.num_in_file_negatives = num_in_file_negatives
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.prebuilt_data_path = prebuilt_data_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if prebuilt_corpus_path is not None:
            logger.info(f"Loading prebuilt corpus from {prebuilt_corpus_path}")
            with open(prebuilt_corpus_path, "rb") as f:
                c = pickle.load(f)
                if type(c) == Corpus:
                    self.corpus = c
                elif type(c) == IndexedCorpus:
                    self.corpus = c.corpus
                    self.corpus_embeddings = c.embeddings
                else:
                    raise TypeError("Prebuilt corpus expects an IndexedCorpus or Corpus object.")

        else:
            self.corpus = Corpus(corpus_path)
            self.corpus_embeddings = None
            self.embeddings_staled = True

        metadata = json.load(open(os.path.join(data_path, "../metadata.json")))
        repo = LeanGitRepo(**metadata["from_repo"])

    def prepare_data(self) -> None:
        pass

    def _load_or_build_dataset(
        self, 
        split: str, 
        is_train: bool, 
        data_paths: List[str]
    ) -> RetrievalDataset:
        """Load prebuilt dataset if available, otherwise build from scratch."""
        if self.prebuilt_data_path:
            pkl_path = os.path.join(self.prebuilt_data_path, f"{split}_dataset.pkl")
            logger.info(f"Loading prebuilt {split} dataset from {pkl_path}")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            
            # Create dataset instance without calling __init__
            ds = RetrievalDataset.__new__(RetrievalDataset)
            ds.corpus = self.corpus
            ds.num_negatives = self.num_negatives
            ds.num_in_file_negatives = self.num_in_file_negatives
            ds.max_seq_len = self.max_seq_len
            ds.tokenizer = self.tokenizer
            ds.is_train = is_train
            ds.data = data
            return ds
        else:
            return RetrievalDataset(
                data_paths,
                self.corpus,
                self.num_negatives,
                self.num_in_file_negatives,
                self.max_seq_len,
                self.tokenizer,
                is_train,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = self._load_or_build_dataset(
            "train",
            is_train=True,
            data_paths=[os.path.join(self.data_path, "train.json")]
        )

        if stage in (None, "fit", "validate"):
            self.ds_val = self._load_or_build_dataset(
                "val",
                is_train=False,
                data_paths=[os.path.join(self.data_path, "val.json")]
            )

        if stage in (None, "fit", "predict"):
            # For ds_pred, we need to combine train, val, and test
            if self.prebuilt_data_path:
                # Load the three separate prebuilt datasets and combine them
                ds_pred = RetrievalDataset.__new__(RetrievalDataset)
                ds_pred.corpus = self.corpus
                ds_pred.num_negatives = self.num_negatives
                ds_pred.num_in_file_negatives = self.num_in_file_negatives
                ds_pred.max_seq_len = self.max_seq_len
                ds_pred.tokenizer = self.tokenizer
                ds_pred.is_train = False
                
                # Load and combine all three splits
                combined_data = []
                for split in ["train", "val", "test"]:
                    pkl_path = os.path.join(self.prebuilt_data_path, f"{split}_dataset.pkl")
                    logger.info(f"Loading prebuilt {split} dataset for prediction from {pkl_path}")
                    with open(pkl_path, "rb") as f:
                        combined_data.extend(pickle.load(f))
                
                ds_pred.data = combined_data
                self.ds_pred = ds_pred
            else:
                # Build from scratch
                self.ds_pred = RetrievalDataset(
                    [
                        os.path.join(self.data_path, f"{split}.json")
                        for split in ("train", "val", "test")
                    ],
                    self.corpus,
                    self.num_negatives,
                    self.num_in_file_negatives,
                    self.max_seq_len,
                    self.tokenizer,
                    is_train=False,
                )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_train.collate,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_val.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_pred,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.ds_pred.collate,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
