# prompt_eval.py
from __future__ import annotations
import os, re, time, uuid, json, hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------- Config / helpers -----------------------------

def pick_device(explicit: Optional[str] = None) -> str:
    """Choose the best available device (MPS for M1/M2, CUDA, else CPU)."""
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def strip_code_fences(text: str) -> str:
    """Remove fenced code blocks ``` ... ``` to avoid spurious number matches."""
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def extract_numbers(text: str, n_max: Optional[int] = None) -> List[float]:
    """Extract floating-point numbers in order of appearance."""
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    nums = [float(x) for x in matches]
    return nums[:n_max] if n_max else nums


def softmax_normalize(log_probs: np.ndarray) -> np.ndarray:
    """Turn a vector of log-probs into normalized weights (stable)."""
    z = log_probs - log_probs.max() # centers the array by subtracting the max number  
    w = np.exp(z) # find the normalised wheight with the exponantial of each number in the list 
    denom = w.sum() # calculates the denomimator which is the normalising constant that we use to get the numbers to sum to 1 
    return w / denom if denom > 0 else np.zeros_like(w) # divides the weights by the sum of exp to get them to sum to 1 if they are over 0 


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


@dataclass
class RunMeta:
    run_id: str
    ts_iso: str
    model_name: str
    prompt_hash: str
    prompt: str
    max_new_tokens: int
    device: str

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ----------------------------- Core runner ----------------------------------

def load_model(model_name: str, device: Optional[str] = None, dtype: str = "float16"):
    """
    Load tokenizer + model. For Apple Silicon, float16 on MPS is ideal.
    dtype can be 'float16' or 'float32'. (bfloat16 not supported on mps.)
    """
    device = pick_device(device)
    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device


@torch.inference_mode()
def generate_with_scores(
    tokenizer,
    model,
    prompt: str,
    device: str,
    max_new_tokens: int = 40,
    do_sample: bool = False,
) -> Dict[str, Any]:
    """
    Generate tokens and return raw outputs plus decoded text.
    Uses return_dict_in_generate + output_scores to compute token metrics.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
    )
    decoded = tokenizer.decode(gen.sequences[0])
    return {
        "inputs": inputs,
        "decoded": decoded,
        "scores": gen.scores,              # list of [batch, vocab] per step
        "sequences": gen.sequences,        # [batch, input+generated]
    }


def token_metrics_from_scores(
    tokenizer,
    outputs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a per-token DataFrame with token, id, log_prob, prob, entropy, weight_norm.
    Only covers the **generated** tokens (not the prompt tokens).
    """
    scores_list = outputs["scores"]
    seq = outputs["sequences"][0]
    input_len = outputs["inputs"]["input_ids"].shape[1]
    gen_ids = seq[input_len:]  # the newly generated token ids

    # stack scores into [T, vocab]
    scores = torch.stack(scores_list, dim=1)[0]  # [T, vocab]
    log_probs = F.log_softmax(scores, dim=-1)    # [T, vocab]
    probs = log_probs.exp()

    rows = []
    # For normalized weights across the generated sequence, we need the chosen-token logprobs:
    chosen_lp = []

    for t, tid in enumerate(gen_ids):
        tid = int(tid.item())
        lp = float(log_probs[t, tid].item())
        pr = float(probs[t, tid].item())
        ent = float((-probs[t] * log_probs[t]).sum().item())  # entropy at step t
        tok = tokenizer.decode([tid])

        rows.append({
            "t_index": t,
            "token_id": tid,
            "token": tok,
            "log_prob": lp,
            "prob": pr,
            "entropy": ent,
        })
        chosen_lp.append(lp)

    # normalized weight across steps (softmax over chosen log-probs)
    chosen_lp = np.array(chosen_lp, dtype=float)
    w = softmax_normalize(chosen_lp)
    for i, wi in enumerate(w):
        rows[i]["weight_norm"] = float(wi)

    return pd.DataFrame(rows)


def analyze_prompt(
    prompt: str,
    model_name: str,
    results_csv: str = "results.csv",
    device: Optional[str] = None,
    max_new_tokens: int = 40,
    do_sample: bool = False,
    n_coeffs_to_extract: Optional[int] = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full pipeline:
      - load model (if not already loaded outside),
      - generate,
      - compute per-token metrics,
      - extract numeric coefficients (after stripping code blocks),
      - append run to CSV with metadata columns.

    Returns:
      df (per-token metrics), info (metadata incl. decoded text and coefficients)
    """
    tok, mdl, device = load_model(model_name, device=device)
    raw = generate_with_scores(tok, mdl, prompt, device, max_new_tokens, do_sample)

    # Per-token metrics for generated tokens
    df = token_metrics_from_scores(tok, raw)

    # Coefficient extraction (strip fenced code first)
    decoded_clean = strip_code_fences(raw["decoded"])
    coeffs = extract_numbers(decoded_clean, n_max=n_coeffs_to_extract) if n_coeffs_to_extract else []

    # Build metadata
    run_id = str(uuid.uuid4())[:8]
    ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta = RunMeta(
        run_id=run_id,
        ts_iso=ts_iso,
        model_name=model_name,
        prompt_hash=hash_prompt(prompt),
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    # Add metadata columns to df
    df.insert(0, "run_id", meta.run_id)
    df.insert(1, "ts_iso", meta.ts_iso)
    df.insert(2, "model_name", meta.model_name)
    df.insert(3, "prompt_hash", meta.prompt_hash)

    # Add decoded output and coefficients as tail metadata (one row per token; keep separate dict too)
    df["decoded_full"] = raw["decoded"]
    df["coefficients_json"] = json.dumps(coeffs)

    # Append to CSV
    write_header = (not os.path.exists(results_csv))
    df.to_csv(results_csv, index=False, mode="a", header=write_header)

    info = {
        "run": asdict(meta),
        "decoded": raw["decoded"],
        "coefficients": coeffs,
        "n_generated_tokens": len(df),
        "results_csv": os.path.abspath(results_csv),
    }
    return df, info


# ----------------------------- Example usage --------------------------------
if __name__ == "__main__":
    prompt = (
        "Given that a young healthy man has coefficients age=0.008, chol=0.015, bp=0.020, "
        "adjust them to describe an old woman. Briefly explain how each coefficient should change. "
        "Then, on the final line, write: Final coefficients: age=<value>, chol=<value>, bp=<value>. "
        "Do not write any code or explanations after that."
    )
    df, info = analyze_prompt(
        prompt=prompt,
        model_name="microsoft/phi-3-mini-4k-instruct",  # good on M1
        results_csv="results.csv",
        device=None,                # auto-pick (MPS on Mac)
        max_new_tokens=60,
        do_sample=False,
        n_coeffs_to_extract=3,      # extract first 3 numbers if present
    )
    print("Saved to:", info["results_csv"])
    print("Coefficients:", info["coefficients"])
    print(df.head(8))
