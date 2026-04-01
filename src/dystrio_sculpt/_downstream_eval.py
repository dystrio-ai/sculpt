"""Cheap downstream proxy eval for search SLO.

Runs a fast multiple-choice accuracy check on a small held-out set from
MMLU, HellaSwag, and ARC-Challenge.  MMLU is weighted most heavily because
our data shows it is the most sensitive metric to compression damage
(mean -28%, std 15 — degrades first and hardest).

Designed to replace PPL as the quality signal during Thompson Sampling Search.

Typical runtime: 60-120 seconds for 250 questions on a 3B model (GPU).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_dataset

_log = logging.getLogger(__name__)

N_MMLU = 100
N_HELLASWAG = 80
N_ARC = 70
N_HUMANEVAL = 80
N_MBPP = 70


@dataclass
class DownstreamProbe:
    """Pre-loaded mini evaluation set for fast downstream accuracy checks."""

    questions: List[Dict]
    baseline_accuracy: Optional[float] = None


def _load_mmlu_questions(n: int, seed: int = 42) -> List[Dict]:
    """Load MMLU validation examples as multiple-choice questions.

    Samples across all subjects to get broad coverage of knowledge domains.
    """
    ds = load_dataset("cais/mmlu", "all", split="validation")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]

    questions = []
    for idx in indices:
        row = ds[idx]
        ctx = f"Question: {row['question']}\nChoices:\n"
        choices = row["choices"]
        for i, c in enumerate(choices):
            ctx += f"  ({chr(65+i)}) {c}\n"
        ctx += "Answer: ("

        label = int(row["answer"])
        choice_tokens = [f"{chr(65+i)})" for i in range(len(choices))]
        questions.append({
            "task": "mmlu",
            "context": ctx,
            "choices": choice_tokens,
            "label": label,
            "subject": row.get("subject", ""),
        })
    return questions


def _load_hellaswag_questions(n: int, seed: int = 42) -> List[Dict]:
    """Load HellaSwag validation examples as multiple-choice questions."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]

    questions = []
    for idx in indices:
        row = ds[idx]
        ctx = row["activity_label"] + ": " + row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        questions.append({
            "task": "hellaswag",
            "context": ctx,
            "choices": endings,
            "label": label,
        })
    return questions


def _load_arc_questions(n: int, seed: int = 42) -> List[Dict]:
    """Load ARC-Challenge validation examples as multiple-choice questions."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]

    questions = []
    for idx in indices:
        row = ds[idx]
        ctx = "Question: " + row["question"] + "\nAnswer:"
        choices_list = row["choices"]
        choice_texts = choices_list["text"]
        answer_key = row["answerKey"]
        labels = choices_list["label"]
        try:
            label = labels.index(answer_key)
        except ValueError:
            continue
        questions.append({
            "task": "arc",
            "context": ctx,
            "choices": [" " + c for c in choice_texts],
            "label": label,
        })
    return questions


def _load_humaneval_questions(n: int, seed: int = 42) -> List[Dict]:
    """Load HumanEval test examples as code completion questions.

    Each question presents a function signature + docstring and asks the model
    to pick the correct completion from a set of candidates (the canonical
    solution vs plausible distractors derived from the solution).
    """
    ds = load_dataset("openai/openai_humaneval", "openai_humaneval", split="test")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]

    questions = []
    for idx in indices:
        row = ds[idx]
        prompt = row["prompt"]
        solution = row["canonical_solution"]

        lines = [l for l in solution.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        correct = solution.strip()
        mid = max(1, len(lines) // 2)
        distractor_a = "\n".join(lines[:mid]).strip()
        distractor_b = "\n".join(lines[mid:]).strip()
        distractor_c = "\n".join(reversed(lines)).strip()

        choices = [correct, distractor_a, distractor_b, distractor_c]
        label = 0
        perm = list(range(4))
        rng.shuffle(perm)
        choices = [choices[i] for i in perm]
        label = perm.index(0)

        questions.append({
            "task": "humaneval",
            "context": prompt,
            "choices": ["\n" + c for c in choices],
            "label": label,
            "task_id": row.get("task_id", ""),
        })
    return questions


def _load_mbpp_questions(n: int, seed: int = 42) -> List[Dict]:
    """Load MBPP examples as code generation questions.

    Presents the task description and asks the model to pick the correct
    code solution from candidates.
    """
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]

    questions = []
    for idx in indices:
        row = ds[idx]
        desc = row.get("text", row.get("prompt", ""))
        code = row.get("code", "")
        if not desc or not code:
            continue

        ctx = f"# Task: {desc}\n"
        correct = code.strip()

        lines = [l for l in correct.split("\n") if l.strip()]
        if len(lines) < 2:
            continue

        mid = max(1, len(lines) // 2)
        distractor_a = "\n".join(lines[:mid]).strip()
        distractor_b = "\n".join(reversed(lines)).strip()
        distractor_c = correct.replace("return", "yield", 1).strip()

        choices = [correct, distractor_a, distractor_b, distractor_c]
        label = 0
        perm = list(range(4))
        rng.shuffle(perm)
        choices = [choices[i] for i in perm]
        label = perm.index(0)

        questions.append({
            "task": "mbpp",
            "context": ctx,
            "choices": ["\n" + c for c in choices],
            "label": label,
        })
    return questions


# Code-related workload prefixes that trigger code-specific downstream eval.
_CODE_WORKLOADS = frozenset({"code", "code_v1", "code_v2", "code_starcoder"})


def load_downstream_probe(
    n_mmlu: int = N_MMLU,
    n_hellaswag: int = N_HELLASWAG,
    n_arc: int = N_ARC,
    n_humaneval: int = N_HUMANEVAL,
    n_mbpp: int = N_MBPP,
    seed: int = 42,
    workload: Optional[str] = None,
) -> DownstreamProbe:
    """Load the mini eval set.  Call once at search startup.

    When *workload* is a code-specific preset, HumanEval and MBPP questions
    are added (and weighted) to reflect code-generation quality.
    """
    is_code = workload in _CODE_WORKLOADS if workload else False

    loaders: List[Tuple[object, int, str]] = [
        (_load_mmlu_questions, n_mmlu, "mmlu"),
        (_load_hellaswag_questions, n_hellaswag, "hellaswag"),
        (_load_arc_questions, n_arc, "arc"),
    ]
    if is_code:
        loaders.append((_load_humaneval_questions, n_humaneval, "humaneval"))
        loaders.append((_load_mbpp_questions, n_mbpp, "mbpp"))

    total_str = " + ".join(f"{n} {name.upper()}" for _, n, name in loaders)
    _log.info("loading downstream probe: %s questions", total_str)

    qs: List[Dict] = []
    for loader, n, name in loaders:
        try:
            loaded = loader(n, seed)
            qs.extend(loaded)
            _log.info("  %s: %d questions loaded", name, len(loaded))
        except Exception as exc:
            _log.warning("failed to load %s: %s", name, exc)

    if not qs:
        _log.warning("no downstream probe questions loaded — SLO will fall back to PPL")

    return DownstreamProbe(questions=qs)


@torch.no_grad()
def _score_choice(
    model, tokenizer, context: str, choice: str,
    device: str, max_len: int = 512,
) -> float:
    """Log-likelihood of *choice* given *context*, length-normalized."""
    full_text = context + choice
    ctx_enc = tokenizer(context, return_tensors="pt", truncation=True, max_length=max_len)
    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_len)
    full_ids = full_enc["input_ids"].to(device)
    ctx_len = ctx_enc["input_ids"].shape[1]
    choice_len = full_ids.shape[1] - ctx_len
    if choice_len <= 0:
        return float("-inf")

    outputs = model(full_ids)
    logits = outputs.logits[0]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    token_lps = log_probs[ctx_len - 1 : full_ids.shape[1] - 1]
    target_ids = full_ids[0, ctx_len:]
    gathered = token_lps.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    return float(gathered.sum().item()) / choice_len


# MMLU is weighted 2x for general workloads.  HumanEval/MBPP get 2x for code
# workloads — the weight map is the same for all workloads so that code
# benchmarks naturally dominate when present.
_TASK_WEIGHTS = {
    "mmlu": 2.0, "hellaswag": 1.0, "arc": 1.0,
    "humaneval": 2.0, "mbpp": 2.0,
}


@torch.no_grad()
def eval_downstream_accuracy(
    model,
    tokenizer,
    probe: DownstreamProbe,
    device: str,
) -> Dict[str, float]:
    """Run the mini downstream eval.  Returns per-task and weighted accuracy.

    MMLU is weighted 2x in the aggregate score because it's the most
    sensitive to compression damage and the primary enterprise benchmark.
    """
    model.eval()
    if not probe.questions:
        return {"accuracy": 0.0, "n_questions": 0}

    per_task: Dict[str, List[bool]] = {}

    for q in probe.questions:
        scores = [
            _score_choice(model, tokenizer, q["context"], c, device)
            for c in q["choices"]
        ]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        hit = pred == q["label"]
        per_task.setdefault(q["task"], []).append(hit)

    result: Dict[str, float] = {"n_questions": sum(len(v) for v in per_task.values())}
    for task, hits in per_task.items():
        result[f"{task}_accuracy"] = sum(hits) / max(1, len(hits))

    # Weighted aggregate: MMLU counts 2x
    weighted_sum = 0.0
    weight_total = 0.0
    for task, hits in per_task.items():
        w = _TASK_WEIGHTS.get(task, 1.0)
        weighted_sum += w * (sum(hits) / max(1, len(hits)))
        weight_total += w
    result["accuracy"] = weighted_sum / max(1e-9, weight_total)

    return result
