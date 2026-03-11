"""Load datasets for training (clustering/profiling) and benchmarking.

Training data: SQuAD 2.0, FinQA (SEC 10-K), CoQA
Benchmark-only: FinanceBench (Patronus AI, 150 samples)
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import load_dataset

from config import PROMPTS_PER_BENCHMARK, DATA_DIR, RANDOM_SEED


def load_squad2(n: int) -> list[dict]:
    """Load n questions from SQuAD 2.0 validation set (answerable + unanswerable)."""
    ds = load_dataset("rajpurkar/squad_v2", split="validation")

    random.seed(RANDOM_SEED)
    sampled = random.sample(list(ds), min(n, len(ds)))

    prompts = []
    for i, ex in enumerate(sampled):
        context = ex["context"]
        question = ex["question"]
        prompt_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        # Unanswerable questions have empty answer lists
        if len(ex["answers"]["text"]) > 0:
            ground_truth = ex["answers"]["text"][0]
        else:
            ground_truth = "unanswerable"

        prompts.append({
            "id": f"squad2_{i:03d}",
            "benchmark": "squad2",
            "text": prompt_text,
            "ground_truth": ground_truth,
            "context": context,
            "question": question,
        })

    return prompts



def load_finqa(n: int) -> list[dict]:
    """Load n questions from financial QA over SEC 10-K filings."""
    ds = load_dataset("virattt/financial-qa-10K", split="train")

    random.seed(RANDOM_SEED + 2)
    indices = random.sample(range(len(ds)), min(n, len(ds)))
    sampled = [ds[i] for i in indices]

    prompts = []
    for i, ex in enumerate(sampled):
        context = ex["context"]
        question = ex["question"]
        prompt_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        prompts.append({
            "id": f"finqa_{i:03d}",
            "benchmark": "finqa",
            "text": prompt_text,
            "ground_truth": ex["answer"],
            "context": context,
            "question": question,
        })

    return prompts


def load_coqa(n: int) -> list[dict]:
    """Load n questions from CoQA, flattening multi-turn conversations into individual prompts.

    Each turn becomes its own prompt with the conversation history included.
    """
    ds = load_dataset("stanfordnlp/coqa", split="train")

    random.seed(RANDOM_SEED + 4)
    sampled = random.sample(list(ds), min(n * 2, len(ds)))  # oversample since we flatten

    prompts = []
    for ex in sampled:
        context = ex["story"]
        questions = ex["questions"]
        answers = ex["answers"]["input_text"]

        for turn_idx in range(len(questions)):
            # Build conversation history from prior turns
            history = ""
            for prev in range(turn_idx):
                history += f"Q: {questions[prev]}\nA: {answers[prev]}\n"

            current_q = questions[turn_idx]
            if history:
                prompt_text = (
                    f"Context: {context}\n\n"
                    f"Conversation history:\n{history}\n"
                    f"Question: {current_q}\n\nAnswer:"
                )
            else:
                prompt_text = f"Context: {context}\n\nQuestion: {current_q}\n\nAnswer:"

            prompts.append({
                "id": f"coqa_{len(prompts):04d}",
                "benchmark": "coqa",
                "text": prompt_text,
                "ground_truth": answers[turn_idx],
                "context": context,
                "question": current_q,
            })

            if len(prompts) >= n:
                break

        if len(prompts) >= n:
            break

    return prompts[:n]


def load_financebench() -> list[dict]:
    """Load all 150 FinanceBench samples (benchmark-only, not used for training).

    Uses evidence_text as context since the dataset references SEC filings.
    """
    ds = load_dataset("PatronusAI/financebench", split="train")

    prompts = []
    for i, ex in enumerate(ds):
        # Combine all evidence texts as context
        evidence_parts = []
        for ev in ex["evidence"]:
            evidence_parts.append(ev["evidence_text"])
        context = "\n\n".join(evidence_parts)

        question = ex["question"]
        prompt_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"

        prompts.append({
            "id": f"financebench_{i:03d}",
            "benchmark": "financebench",
            "text": prompt_text,
            "ground_truth": ex["answer"],
            "context": context,
            "question": question,
        })

    return prompts


def main():
    os.makedirs(str(DATA_DIR), exist_ok=True)

    print(f"Loading SQuAD 2.0 ({PROMPTS_PER_BENCHMARK} samples)...")
    squad = load_squad2(PROMPTS_PER_BENCHMARK)
    squad_path = os.path.join(str(DATA_DIR), "squad2_subset.json")
    with open(squad_path, "w") as f:
        json.dump(squad, f, indent=2)
    print(f"  Saved {len(squad)} prompts to {squad_path}")

    print(f"Loading FinQA / SEC 10-K ({PROMPTS_PER_BENCHMARK} samples)...")
    finqa = load_finqa(PROMPTS_PER_BENCHMARK)
    finqa_path = os.path.join(str(DATA_DIR), "finqa_subset.json")
    with open(finqa_path, "w") as f:
        json.dump(finqa, f, indent=2)
    print(f"  Saved {len(finqa)} prompts to {finqa_path}")

    print(f"Loading CoQA ({PROMPTS_PER_BENCHMARK} samples)...")
    coqa = load_coqa(PROMPTS_PER_BENCHMARK)
    coqa_path = os.path.join(str(DATA_DIR), "coqa_subset.json")
    with open(coqa_path, "w") as f:
        json.dump(coqa, f, indent=2)
    print(f"  Saved {len(coqa)} prompts to {coqa_path}")

    total = len(squad) + len(finqa) + len(coqa)
    print(f"\nTraining data: {total} prompts ready for grid search.")

    # Benchmark-only dataset
    print(f"\nLoading FinanceBench (all 150 samples, benchmark-only)...")
    fb = load_financebench()
    fb_path = os.path.join(str(DATA_DIR), "financebench_subset.json")
    with open(fb_path, "w") as f:
        json.dump(fb, f, indent=2)
    print(f"  Saved {len(fb)} prompts to {fb_path}")

    print(f"\nGrand total: {total + len(fb)} prompts ({total} training + {len(fb)} benchmark-only).")


if __name__ == "__main__":
    main()
