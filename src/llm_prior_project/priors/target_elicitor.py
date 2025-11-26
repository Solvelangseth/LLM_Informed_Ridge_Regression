import os
from dotenv import load_dotenv 
from typing import List, Dict, Optional
import openai
import pandas as pd 
import json
import re
import time
import hashlib
from pathlib import Path
import numpy as np


class LLMTargetElicitor:
    """
    Handles LLM communication for coefficient target elicitation.
    Uses robust plaintext extraction + confidence weighting.
    JSON is used only for metadata storage.
    """

    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        self.model_name = model_name 
        self.api_key = api_key or self._get_api_key_from_env()

        self.supported_models = {
            'gpt-4': 'openai',
            'gpt-3.5-turbo': 'openai',
            'claude-3-sonnet': 'anthropic',
            'claude-3-haiku': 'anthropic'
        }

        if model_name not in self.supported_models:
            raise ValueError(f"Unsupported model {model_name}")

        self.api_provider = self.supported_models[model_name]

    # ------------------------------------------------------
    def _get_api_key_from_env(self) -> Optional[str]:
        load_dotenv()  
        for key_name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
            v = os.getenv(key_name)
            if v:
                return v
        return None

    # ------------------------------------------------------
    def call_llm_api(self, prompt: str) -> Optional[str]:
        """Universal LLM call used everywhere."""
        try:
            if self.api_provider == 'openai':
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                return response.choices[0].message.content

            elif self.api_provider == 'anthropic':
                import anthropic
                client = anthropic.Anthropic(api_key=self.api_key)
                response = client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

        except Exception as e:
            print(f"API call failed: {e}")
            return None

    # ------------------------------------------------------
    def parse_plaintext_targets(self, response_text: str, feature_names: List[str]):
        """
        Parse robust plaintext format:
        
        feature1: 0.12
        feature2: -1.3
        feature3: 0.44
        confidence: 0.82
        """

        if not response_text:
            return None

        text = response_text.lower()
        targets = []

        # Extract each feature value
        for f in feature_names:
            pattern = rf"{f.lower()}\s*:\s*([-+]?\d*\.\d+|\d+)"
            m = re.search(pattern, text)
            if not m:
                return None
            targets.append(float(m.group(1)))

        # Extract confidence
        cm = re.search(r"confidence\s*:\s*([-+]?\d*\.\d+|\d+)", text)
        if cm:
            conf = float(cm.group(1))
            conf = max(0.0, min(1.0, conf))
        else:
            conf = 1.0

        return {
            "targets": targets,
            "confidence": conf,
            "raw_response": response_text
        }

    # ------------------------------------------------------
    def run_rephrased_batch(self, prompt: str, feature_names: List[str],
                            n_variants: int, out_path: str):
        """
        Generate N rephrasings, extract robust plaintext targets,
        compute stats, and save metadata JSON.
        """

        if not prompt:
            return None

        batch_id = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")

        all_targets = []
        all_conf = []
        variants = []

        # -------------------------
        for i in range(n_variants):

            # 1. Rephrase freely but preserve meaning
            rephrase_instruction = (
                "Rephrase this prompt while preserving meaning. "
                "Change wording only:\n\n"
                f"{prompt}"
            )
            rephrased = self.call_llm_api(rephrase_instruction)
            if not rephrased:
                continue

            # 2. Extraction instruction using plaintext format
            feature_lines = "\n".join(
                f"{fname}: <number>" for fname in feature_names
            )

            extraction_prompt = (
                f"{rephrased}\n\n"
                "Respond ONLY in the following exact plaintext format:\n"
                f"{feature_lines}\n"
                "confidence: <number between 0 and 1>\n"
                "\n"
                "No extra text."
            )

            # 3. Call LLM for targets
            resp = self.call_llm_api(extraction_prompt)
            if not resp:
                continue

            # 4. Parse plaintext
            parsed = self.parse_plaintext_targets(resp, feature_names)
            if not parsed:
                continue

            targets = parsed["targets"]
            conf = parsed["confidence"]

            all_targets.append(targets)
            all_conf.append(conf)

            variants.append({
                "variant_id": f"{batch_id}_{i}",
                "prompt_rephrased": rephrased,
                "targets": targets,
                "confidence": conf,
                "raw_response": resp
            })

        # -------------------------
        # Compute averages, variance, std
        avg_unweighted = None
        avg_weighted = None
        targets_var = None
        targets_std = None

        if all_targets:
            arr = np.array(all_targets, float)
            conf_arr = np.array(all_conf, float)

            avg_unweighted = arr.mean(axis=0).tolist()
            targets_var = arr.var(axis=0).tolist()
            targets_std = arr.std(axis=0).tolist()

            if conf_arr.sum() > 0:
                avg_weighted = (
                    (arr * conf_arr[:, None]).sum(axis=0) / conf_arr.sum()
                ).tolist()

        # -------------------------
        # Metadata JSON (not used in computation)
        results = {
            "batch_id": batch_id,
            "timestamp": ts,
            "model_name": self.model_name,
            "original_prompt": prompt,
            "n_variants": n_variants,
            "variants": variants,
            "avg_targets_unweighted": avg_unweighted,
            "avg_targets_weighted": avg_weighted,
            "targets_var": targets_var,
            "targets_std": targets_std
        }

        # Save JSON metadata
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Saved batch → {path}")
        return results
