"""
Benchmark: separate tight-padded calls vs one fused padded call
for google/byt5-small (encoder only, eager attention).

Demonstrates the padding penalty due to O(n²) attention on
character-level sequences with highly variable lengths.

Run on a GPU node:
  srun --partition=a100-galvani --gres=gpu:1 --time=00:10:00 \
       python test_fa2_padding.py
"""

import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForTextEncoding, AutoTokenizer

MODEL_NAME = "google/byt5-small"
WARMUP_ITERS = 10
BENCH_ITERS = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode(model, input_ids, attention_mask):
    """Mean-pooled, L2-normalised encoding (same as PremiseRetriever._encode)."""
    hidden = model(input_ids=input_ids, attention_mask=attention_mask,
                   return_dict=True).last_hidden_state
    lens = attention_mask.sum(dim=1)
    features = (hidden * attention_mask.unsqueeze(2)).sum(dim=1) / lens.unsqueeze(1)
    return F.normalize(features, dim=1)


def tokenize_group(tokenizer, texts, max_len=1024):
    return tokenizer(texts, padding="longest", max_length=max_len,
                     truncation=True, return_tensors="pt")


def pad_to(ids, mask, target_len):
    pad_len = target_len - ids.size(1)
    if pad_len > 0:
        ids = F.pad(ids, (0, pad_len), value=0)
        mask = F.pad(mask, (0, pad_len), value=0)
    return ids, mask


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(label, fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters * 1000
    print(f"  {label:45s}  {elapsed:7.2f} ms/iter")
    return elapsed


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ---- Synthetic data mimicking training batches ----
    # Contexts: long (ByT5 is character-level, so these are ~300-500 tokens)
    contexts = [
        "Theorem: For all natural numbers n, the sum 1 + 2 + ... + n equals n*(n+1)/2. " * 4,
        "Lemma: If a group G is abelian and finite, then every subgroup of G is normal. Proof follows from definition of normality and commutativity. " * 2,
        "Definition: A topological space X is compact if every open cover has a finite subcover. We prove this for metric spaces using sequential compactness. " * 3,
    ]
    # Premises: short (30-100 character-tokens)
    pos_premises = [
        "Nat.add_comm",
        "Group.normal_of_abelian (G : Group)",
        "TopologicalSpace.compact_iff_seq_compact",
    ]
    neg_groups = [
        ["Nat.mul_comm", "Ring.zero_mul (R : Ring)", "Metric.dist_triangle"],
        ["Nat.succ_pos", "Field.inv_cancel", "Set.finite_union"],
        ["List.length_append", "Fin.val_last", "Real.sqrt_nonneg"],
    ]

    # Tokenize each group with its own tight padding
    tok_ctx = tokenize_group(tokenizer, contexts)
    tok_pos = tokenize_group(tokenizer, pos_premises)
    tok_negs = [tokenize_group(tokenizer, ng) for ng in neg_groups]

    ctx_ids = tok_ctx.input_ids.to(device)
    ctx_mask = tok_ctx.attention_mask.to(device)
    pos_ids = tok_pos.input_ids.to(device)
    pos_mask = tok_pos.attention_mask.to(device)
    neg_ids_list = [t.input_ids.to(device) for t in tok_negs]
    neg_mask_list = [t.attention_mask.to(device) for t in tok_negs]

    all_groups_ids = [ctx_ids, pos_ids] + neg_ids_list
    all_groups_mask = [ctx_mask, pos_mask] + neg_mask_list

    # Pre-compute the fused padded batch (all groups padded to global max)
    max_len = max(t.size(1) for t in all_groups_ids)
    fused_ids_list, fused_mask_list = [], []
    for ids, mask in zip(all_groups_ids, all_groups_mask):
        ids_p, mask_p = pad_to(ids, mask, max_len)
        fused_ids_list.append(ids_p)
        fused_mask_list.append(mask_p)
    fused_ids = torch.cat(fused_ids_list, dim=0)
    fused_mask = torch.cat(fused_mask_list, dim=0)

    # Pre-compute the two-call version (contexts separate, premises fused)
    prem_groups_ids = [pos_ids] + neg_ids_list
    prem_groups_mask = [pos_mask] + neg_mask_list
    max_prem_len = max(t.size(1) for t in prem_groups_ids)
    prem_ids_list, prem_mask_list = [], []
    for ids, mask in zip(prem_groups_ids, prem_groups_mask):
        ids_p, mask_p = pad_to(ids, mask, max_prem_len)
        prem_ids_list.append(ids_p)
        prem_mask_list.append(mask_p)
    prem_fused_ids = torch.cat(prem_ids_list, dim=0)
    prem_fused_mask = torch.cat(prem_mask_list, dim=0)

    # Compute padding stats
    total_separate = sum(t.size(0) * t.size(1) for t in all_groups_ids)
    total_fused = fused_ids.size(0) * fused_ids.size(1)
    total_two_call = (ctx_ids.size(0) * ctx_ids.size(1) +
                      prem_fused_ids.size(0) * prem_fused_ids.size(1))

    print(f"\n{'='*65}")
    print(f"Model: {MODEL_NAME}  (encoder only, eager attention)")
    print(f"{'='*65}")
    print(f"  Group            batch  seq_len   tokens")
    print(f"  {'─'*50}")
    print(f"  Contexts           {ctx_ids.size(0):3d}    {ctx_ids.size(1):4d}    {ctx_ids.size(0)*ctx_ids.size(1):6d}")
    print(f"  Pos premises       {pos_ids.size(0):3d}    {pos_ids.size(1):4d}    {pos_ids.size(0)*pos_ids.size(1):6d}")
    for i, nids in enumerate(neg_ids_list):
        print(f"  Neg group {i}        {nids.size(0):3d}    {nids.size(1):4d}    {nids.size(0)*nids.size(1):6d}")
    print(f"  {'─'*50}")
    print(f"  Separate total token-slots:   {total_separate:6d}")
    print(f"  Two-call total token-slots:   {total_two_call:6d}  "
          f"({total_two_call/total_separate:.1f}x)")
    print(f"  Fused total token-slots:      {total_fused:6d}  "
          f"({total_fused/total_separate:.1f}x)")
    print(f"{'='*65}\n")

    # ---- Load model ----
    print("Loading model ...")
    model = AutoModelForTextEncoding.from_pretrained(MODEL_NAME).to(device).eval().half()

    # ---- Strategy 1: Separate calls ----
    @torch.no_grad()
    def separate():
        embs = [encode(model, ids, mask)
                for ids, mask in zip(all_groups_ids, all_groups_mask)]
        return torch.cat(embs, dim=0)

    # ---- Strategy 2: Two calls (ctx separate, premises fused) ----
    @torch.no_grad()
    def two_call():
        ctx_emb = encode(model, ctx_ids, ctx_mask)
        prem_emb = encode(model, prem_fused_ids, prem_fused_mask)
        return torch.cat([ctx_emb, prem_emb], dim=0)

    # ---- Strategy 3: Fully fused ----
    @torch.no_grad()
    def fused():
        return encode(model, fused_ids, fused_mask)

    # ---- Run benchmarks ----
    print("--- Benchmarks ---")
    t_sep   = benchmark("Separate calls (5 calls, tight pad)", separate)
    t_two   = benchmark("Two calls (ctx + premises fused)", two_call)
    t_fused = benchmark("Single fused call (all padded to max)", fused)

    # ---- Correctness check ----
    emb_sep = separate()
    emb_two = two_call()
    emb_fused = fused()
    cos_2 = F.cosine_similarity(emb_sep, emb_two, dim=1)
    cos_f = F.cosine_similarity(emb_sep, emb_fused, dim=1)
    print(f"\n  Cosine sim (separate vs two-call):  "
          f"{cos_2.min():.6f} – {cos_2.max():.6f}")
    print(f"  Cosine sim (separate vs fused):     "
          f"{cos_f.min():.6f} – {cos_f.max():.6f}")

    # ---- Summary ----
    print(f"\n{'='*65}")
    print("SUMMARY  (lower is better)")
    print(f"{'='*65}")
    print(f"  {'Strategy':<48s} {'ms/iter':>8s}  {'ratio':>7s}")
    print(f"  {'─'*65}")
    results = [
        ("Separate (5 calls, tight pad)", t_sep),
        ("Two calls (ctx separate, premises fused)", t_two),
        ("Single fused (all padded to ctx length)", t_fused),
    ]
    best = min(r[1] for r in results)
    for label, t in results:
        ratio = t / best
        marker = " ◀ fastest" if abs(t - best) < 0.01 else ""
        print(f"  {label:<48s} {t:7.2f}   {ratio:6.2f}x{marker}")
    print()


if __name__ == "__main__":
    main()
