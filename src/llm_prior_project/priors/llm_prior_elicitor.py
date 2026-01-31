import os
import re
import numpy as np
from dotenv import load_dotenv

import re
import os
import openai
import json 


class LLMPriorElicitor:
    """
    Production-grade elicitor that:
      - Connects to multiple LLM providers
      - Uses your robust JSON/plaintext parser (extract_targets)
      - Returns empirical mean µ and covariance Σ
      - Provides debugging prints to diagnose prompt issues
    """

    def __init__(self, model_name: str, use_responses_api: bool = None):
        load_dotenv()

        self.model_name = model_name

        if use_responses_api is None:
        # Auto-detect: all GPT-4.1 and GPT-5 models require the new API
            if model_name.startswith(("gpt-5", "gpt-4.1")):
                self.use_responses_api = True
            else:
                self.use_responses_api = False
        else:
            self.use_responses_api = use_responses_api

        # Load keys from .env
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_key = os.getenv("GOOGLE_API_KEY")
        self.xai_key = os.getenv("XAI_API_KEY")
        self.hf_key = os.getenv("HUGGINGFACE_API_KEY")

        self.api_provider = self._detect_provider(model_name)

        print(f"[Elicitor] Using model: {model_name} (provider={self.api_provider})")

    # ----------------------------------------------------------
    # Provider detection
    # ----------------------------------------------------------
    def _detect_provider(self, model_name):
        if model_name.startswith("gpt"):
            return "openai"
        if model_name.startswith("claude"):
            return "anthropic"
        if model_name.startswith("gemini"):
            return "google"
        if model_name.startswith("grok"):
            return "xai"
        if model_name.startswith("hf/"):
            return "huggingface"
        raise ValueError(f"Unknown provider/model name: {model_name}")

    def call(self, prompt: str):
        provider = self.api_provider

        try:
            # -------------------------
            # OpenAI (supports both APIs)
            # -------------------------
            if provider == "openai":
                from openai import OpenAI

                try:
                    client = OpenAI(api_key=self.openai_key)

                    # --- NEW Responses API (GPT-4.1, GPT-5.x) ---
                    if self.use_responses_api:
                        resp = client.responses.create(
                            model=self.model_name,
                            input=prompt,
                            reasoning={"effort": "low"},
                            text={"verbosity": "low"},
                            max_output_tokens=1500
                            # IMPORTANT: no temperature
                        )

                        # Ensure we return a string, not None
                        if hasattr(resp, "output_text") and resp.output_text:
                            return resp.output_text
                        else:
                            print("⚠ Responses API returned no output_text")
                            return None

                    # --- OLD Chat Completions API (GPT-4o, GPT-3.5, etc.) ---
                    else:
                        resp = client.chat.completions.create(
                            model=self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            max_completion_tokens=1000,
                            temperature=0.1,
                        )
                        return resp.choices[0].message.content

                except Exception as e:
                    print(f"OpenAI call failed: {e}")
                    return None


            # -------------------------
            # Anthropic
            # -------------------------
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=self.anthropic_key)
                resp = client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                )
                return resp.content[0].text

            # -------------------------
            # Google (Gemini)
            # -------------------------
            elif provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.google_key)
                model = genai.GenerativeModel(self.model_name)
                resp = model.generate_content(prompt)
                return resp.text

            # -------------------------
            # xAI (Grok)
            # -------------------------
            elif provider == "xai":
                import openai
                client = openai.OpenAI(api_key=self.xai_key, base_url="https://api.x.ai/v1")
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1,
                )
                return resp.choices[0].message.content

            # -------------------------
            # HuggingFace Inference API
            # -------------------------
            elif provider == "huggingface":
                import requests
                url = f"https://api-inference.huggingface.co/models/{self.model_name}"
                headers = {"Authorization": f"Bearer {self.hf_key}"}
                payload = {"inputs": prompt}

                resp = requests.post(url, headers=headers, json=payload)
                resp.raise_for_status()

                # HF models may return different shapes; assume text generation
                data = resp.json()
                if isinstance(data, list) and "generated_text" in data[0]:
                    return data[0]["generated_text"]
                return str(data)

            else:
                raise ValueError(f"Unsupported provider: {provider}")

        except Exception as e:
            print(f"API call failed: {e}")
            return None



    def rephrase_prompt(prompt: str):
        api_key = os.getenv("API_KEY")
        client = openai.OpenAI(api_key=api_key)

        messages = [
            {"role": "system", "content": "Rewrite the prompt clearly while keeping all constraints exactly the same."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Rephrase failed: {e}")
            return prompt

    def extract_targets(self, text: str):
        # 1. Prefer fenced blocks with json
        print("\n--- RAW LLM RESPONSE ---")
        print(text)
        print("--- END RESPONSE ---\n")

        fenced_json = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced_json:
            candidate = fenced_json.group(1).strip()
            try:
                data = json.loads(candidate)
                if "targets" in data:
                    return data["targets"]
            except json.JSONDecodeError:
                pass  # Fallback below
        
        # 2. Fallback: any fenced block ``` ... ```
        fenced = re.search(r"```\s*(.*?)```", text, flags=re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
            try:
                data = json.loads(candidate)
                if "targets" in data:
                    return data["targets"]
            except json.JSONDecodeError:
                pass  # Next fallback
        
        # 3. Fallback: try to interpret the entire response as JSON
        try:
            data = json.loads(text)
            if "targets" in data:
                return data["targets"]
        except Exception:
            pass

        # 4. Fallback: capture ANY JSON object (nested allowed)
        inline = re.search(r"\{[\s\S]*\}", text)
        if inline:
            candidate = inline.group(0)
            try:
                data = json.loads(candidate)
                if "targets" in data:
                    return data["targets"]
            except Exception:
                pass

        raise ValueError("No valid JSON with a 'targets' object found.")




    def sampling_llm(self, num_calls: int, base_prompt: str):
        targets = []
        attempts = 0
        max_attempts = num_calls * 5   # allow retries

        while len(targets) < num_calls and attempts < max_attempts:
            attempts += 1
            response = self.call(prompt=base_prompt)

            try:
                parsed = self.extract_targets(response)
                targets.append(parsed)
                print(f"✔ Valid sample {len(targets)}/{num_calls}")
            except Exception as e:
                print(f"✖ Skipping invalid sample: {e}")
                continue

        if len(targets) < 2:
            raise ValueError(f"Not enough valid samples. Only got {len(targets)}.")

        return self._compute_prior_statistics(targets)

    
    def _compute_prior_statistics(self, targets: list, jitter=1e-6):
        """
        targets: list of dicts with identical keys, e.g.
            [{'bmi': 0.5, 'height': 0.3, 'weight': 0.2}, ...]
        Returns:
            means: 1D numpy array
            cov: 2D numpy array (covariance matrix)
        """

        # Extract feature order from the first sample
        keys = list(targets[0].keys())

        # Convert dicts → row vectors
        vectors = [[d[k] for k in keys] for d in targets]

        # Stack into matrix (samples × features)
        M = np.vstack(vectors)

        n_rows = M.shape[0]
        n_cols = M.shape[1]

        # Compute column-wise means
        means = M.mean(axis=0)

        # Compute unbiased covariance using NumPy
        cov = np.cov(M, rowvar=False)

        # Add jitter for numerical stability
        cov = cov + jitter * np.eye(cov.shape[0])


        # ---- 7. Precision matrix (inverse covariance) ----
        precision = np.linalg.pinv(cov)

        return keys, means, cov, precision


    def _average_targets(self, targets: list):
        totals = {}
        for i in targets:
            for k, v in i.items():
                totals[k] = totals.get(k, 0) + v

        # Compute averages
        n = len(targets)
        averages = {k: totals[k] / n for k in totals}

        return(averages)
    



    def extract_coefficients(self, text: str):
        # 1. Prefer fenced blocks with json
        print("\n--- RAW LLM RESPONSE ---")
        print(text)
        print("--- END RESPONSE ---\n")

        fenced_json = re.search(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidate = None

        if fenced_json:
            candidate = fenced_json.group(1).strip()
        
        # 2. Fallback: try to interpret the entire response as JSON
        if candidate is None:
            inline = re.search(r"\{[\s\S]*\}", text)
            if inline:
                candidate = inline.group(0)

        if candidate:
            try:
                data = json.loads(candidate)
                
                # We expect "intercept" and at least one other coefficient
                if "intercept" in data and len(data) > 1:
                    return data
            except json.JSONDecodeError:
                pass

        raise ValueError("No valid JSON with 'intercept' and coefficients found.")
    
    def elicit_single_model(self, base_prompt: str, feature_names: list):
        """
        Calls the LLM once to get a full linear model (intercept + coefficients).
        Returns: beta_prior (np.array), beta0_prior (float)
        """
        response = self.call(prompt=base_prompt)
        
        if response is None:
             raise ValueError("LLM call failed to return a response.")

        parsed_dict = self.extract_coefficients(response)
        
        # Extract intercept
        beta0_prior = float(parsed_dict.pop("intercept"))
        
        # Extract coefficients in the correct feature order
        beta_prior = np.array([float(parsed_dict[c]) for c in feature_names])
        
        print(f"✔ Successfully elicited model: beta0={beta0_prior:.3f}, mean(|beta|)={np.abs(beta_prior).mean():.3f}")

        return beta_prior, beta0_prior






# Example

"""
sample_prompt = 
You are an API, not a conversational assistant.
Your only job is to produce pure JSON and nothing else.
Do NOT include explanations, greetings, summaries, bullet points, or any text outside JSON.

Given these feature names:
- bmi
- height
- weight

Return a JSON object with numeric coefficient targets for a regularised linear model.
Use floating-point numbers without units.

Output must be EXACTLY and ONLY JSON format like this:

{
"targets": {
    "bmi": <number>,
    "height": <number>,
    "weight": <number>
}
}

Replace <number> with actual numeric values. Do not change keys. Do not add text before or after.
"""

"""
elicitor = LLMPriorElicitor("gpt-4o")


#response = elicitor.call(sample_prompt)

#print(response)

average_targets, c_matrix, p_matrix= elicitor.sampling_llm(10, sample_prompt)

print(f"means: {average_targets}") 
print("")
print(f"covariance matrix: {c_matrix}")
print("")
print(f"precision matrix: {p_matrix}")

"""