"""Script for evaluating the premise retriever."""

import os
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, List
from loguru import logger


def _eval(data, preds_map) -> Tuple[float, float, float]:
    R1 = []
    R10 = []
    MRR = []

    for thm in tqdm(data):
        for i, _ in enumerate(thm["traced_tactics"]):
            key = (thm["file_path"], thm["full_name"], tuple(thm["start"]), i)
            if key not in preds_map:
                continue  # Skip if prediction not available
            
            pred = preds_map[key]
            all_pos_premises = set(pred["all_pos_premises"])
            if len(all_pos_premises) == 0:
                continue

            retrieved_premises = pred["retrieved_premises"]
            TP1 = retrieved_premises[0] in all_pos_premises
            R1.append(float(TP1) / len(all_pos_premises))
            TP10 = len(all_pos_premises.intersection(retrieved_premises[:10]))
            R10.append(float(TP10) / len(all_pos_premises))

            for j, p in enumerate(retrieved_premises):
                if p in all_pos_premises:
                    MRR.append(1.0 / (j + 1))
                    break
            else:
                MRR.append(0.0)

    if len(R1) == 0:
        return 0.0, 0.0, 0.0
    
    R1 = 100 * np.mean(R1)
    R10 = 100 * np.mean(R10)
    MRR = np.mean(MRR)
    return R1, R10, MRR



parser = argparse.ArgumentParser(
    description="Script for evaluating the premise retriever."
)
parser.add_argument(
    "--preds-file",
    type=str,
    required=True,
    help="Path to the retriever's predictions file.",
)
parser.add_argument(
    "--data-path",
    type=str,
    required=True,
    help="Path to the directory containing the train/val/test splits.",
)
parser.add_argument(
    "--splits",
    type=str,
    nargs="+",
    default=["train", "val", "test"],
    help="Which splits to evaluate on (default: all three)",
)
parser.add_argument(
    "--custom-ds",
    type=str,
    help="Path to a custom dataset file.",
)
args = parser.parse_args()
logger.info(args)

logger.info(f"Loading predictions from {args.preds_file}")
preds = pickle.load(open(args.preds_file, "rb"))
preds_map = {
    (p["file_path"], p["full_name"], tuple(p["start"]), p["tactic_idx"]): p
    for p in preds
}
assert len(preds) == len(preds_map), "Duplicate predictions found!"

data_files = []
if args.custom_ds is not None:
    data_files.append(args.custom_ds)
else:
    for split in args.splits:
        data_path = os.path.join(args.data_path, f"{split}.json")
        if not os.path.exists(data_path):
            logger.warning(f"Split file not found: {data_path}, skipping")
            continue
        data_files.append(data_path)
    
for data_file in data_files:
    data = json.load(open(data_file))
    logger.info(f"Evaluating on {data_file}")
    R1, R10, MRR = _eval(data, preds_map)
    if R1 == 0.0 and R10 == 0.0 and MRR == 0.0:
        logger.warning(f"No matching predictions found for {data_file}")
    else:
        logger.info(f"R@1 = {R1:.2f}%, R@10 = {R10:.2f}%, MRR = {MRR:.4f}")