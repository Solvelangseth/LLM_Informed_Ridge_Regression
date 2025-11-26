from __future__ import annotations 
import os, sys, time, json, uuid, hashlib, re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def pick_device(explicit_device: Optional[str] = None) -> str:
  if explicit_device:
    return explicit_device
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