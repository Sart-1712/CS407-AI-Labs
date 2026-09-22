"""Minimal chat-completions client for the coding LLMs used in this lab, with a response cache.

Responses are cached in outputs/llm_responses.json keyed by (model, system prompt, prompt), so the
notebook re-runs identically without network access or credentials.
"""
import hashlib
import json
import os
import pathlib

import requests

SYSTEM = ("You are an expert Python programmer working with discrete "
          "Bayesian networks and the current pgmpy API. Follow the "
          "user specification exactly. Return concise executable code "
          "when code is requested.")

# model name -> (endpoint, model id at the endpoint, env var holding the key, fallback key file)
_ROUTES = {
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": ("https://router.huggingface.co/v1/chat/completions",
                                         "Qwen/Qwen2.5-Coder-1.5B-Instruct:featherless-ai",
                                         "HF_TOKEN", "~/.cache/huggingface/token"),
    "Qwen/Qwen2.5-Coder-32B-Instruct": ("https://openrouter.ai/api/v1/chat/completions",
                                        "qwen/qwen-2.5-coder-32b-instruct",
                                        "OPENROUTER_API_KEY", "~/.config/openrouter/key"),
    "Qwen/Qwen3-Coder-30B-A3B-Instruct": ("https://openrouter.ai/api/v1/chat/completions",
                                          "qwen/qwen3-coder-30b-a3b-instruct",
                                          "OPENROUTER_API_KEY", "~/.config/openrouter/key"),
}


def cache_key(model, prompt):
    return hashlib.sha256(f"{model}\n{SYSTEM}\n{prompt}".encode()).hexdigest()[:16]


def make_ask(cache_path):
    cache_path = pathlib.Path(cache_path)

    def ask(prompt, model, max_tokens=1400):
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        key = cache_key(model, prompt)
        if key in cache:
            return cache[key]["response"]
        url, model_id, env_var, key_file = _ROUTES[model]
        secret = os.environ.get(env_var) or pathlib.Path(key_file).expanduser().read_text().strip()
        r = requests.post(url, headers={"Authorization": f"Bearer {secret}"}, timeout=300,
                          json={"model": model_id, "temperature": 0, "max_tokens": max_tokens, "stream": False,
                                "messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": prompt}]})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        cache[key] = {"model": model, "prompt": prompt, "response": text}
        cache_path.write_text(json.dumps(cache, indent=1))
        return text

    return ask
