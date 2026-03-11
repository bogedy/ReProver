#!/usr/bin/env python3
"""
Generate a 2x2 results table comparing Gemma vs Retriever models,
with and without augmentation. Shows R@1/R@10 in each cell.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_results(base_dir: str = "out") -> dict:
    """Load eval.jsonl files from all experiment directories."""
    experiments = [
        "predict_gemma_val100",
        "predict_gemma_val100_augmented",
        "predict_retriever_val100",
        "predict_retriever_val100_augmented",
    ]
    
    results = {}
    for exp in experiments:
        eval_path = Path(base_dir) / exp / "eval.jsonl"
        with open(eval_path) as f:
            results[exp] = json.load(f)
    
    return results


def create_table(results: dict, output_path: str = "results_table.png"):
    """Create a matplotlib table with R@1/R@10 results."""
    
    # Format cell values as R@1/R@10
    def fmt(exp_name):
        r1 = results[exp_name]["R@1"]
        r10 = results[exp_name]["R@10"]
        return f"{r1:.3f}/{r10:.3f}"
    
    cell_text = [
        [fmt("predict_gemma_val100"), fmt("predict_gemma_val100_augmented")],
        [fmt("predict_retriever_val100"), fmt("predict_retriever_val100_augmented")],
    ]
    
    row_labels = ["Gemma", "Retriever"]
    col_labels = ["Base", "Augmented"]
    
    # Create figure and table
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis("off")
    
    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    
    # Set the upper-left corner cell (between row/col labels) to "(pre"
    table[0, 0].get_text().set_text("(pre")
    
    # Style the table with proper borders
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(1.5)
        cell.set_height(0.15)
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved table to {output_path}")
    plt.show()


if __name__ == "__main__":
    results = load_results()
    create_table(results)
