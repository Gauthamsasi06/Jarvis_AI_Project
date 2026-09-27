"""
JARVIS AI - Agent Layer
Owner: AI Engineer
Wraps multiple LLM providers behind one interface so the rest of the app
doesn't care which model is actually answering. Supports:
  - Anthropic (Claude)               - native SDK
  - Groq (Llama 3, Mixtral, Gemma)   - OpenAI-compatible endpoint, free tier
  - Ollama (local, fully open-source)- OpenAI-compatible endpoint, no key needed
  - OpenRouter / Together AI         - OpenAI-compatible, works the same as Groq

Two jobs, regardless of provider:
  1. insight_agent -> turns a statistical summary into plain-English findings
  2. code_agent     -> turns a natural-language question into pandas code,
                        which the backend then executes against the uploaded data
"""

import re
import io
import contextlib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Default model per provider - override in the UI if you want a different one
DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.1",
    "openrouter": "meta-llama/llama-3.1-70b-instruct",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

# OpenAI-compatible base URLs for each open-source provider
BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
}

INSIGHT_SYSTEM_PROMPT = """You are JARVIS AI, an autonomous data analyst.
You are given a statistical summary of a dataset. Reverse-engineer what is
really happening in the data: name the strongest relationships, call out
anything surprising or risky (outliers, skew, missing data), and suggest
one concrete next step. Write 4-6 short bullet points. No preamble,
no markdown headers, just bullets starting with '- '."""

CODE_SYSTEM_PROMPT = """You are JARVIS AI's code-generation agent.
You write short pandas/plotly code to answer a question about a dataframe
called `df` that is already loaded in memory. Rules:
- Assign the final answer to a variable called `result`.
- If a chart is the best answer, build it with plotly express/graph_objects
  and assign the figure object to `result` (do not call .show()).
- Only use pandas (pd), numpy (np) and plotly (px, go). No file/network/OS access.
- Return ONLY a python code block, nothing else.
"""


# Preset model choices per provider, shown in a dropdown (+ a manual "Custom" entry)
MODEL_PRESETS = {
    "anthropic": ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
    "ollama": ["llama3.1", "llama3", "mistral", "phi3"],
    "openrouter": ["meta-llama/llama-3.1-70b-instruct", "openai/gpt-4o", "anthropic/claude-3.5-sonnet", "qwen/qwen-2.5-72b-instruct"],
    "together": ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "Qwen/Qwen2.5-72B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3"],
}


def _chat(provider: str, api_key: str, model: str, system: str, user: str) -> str:
    """Send one system+user turn to whichever provider is selected, return the text reply."""
    provider = provider.lower()

    if provider == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model, max_tokens=700, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    # Everything else (Groq, Ollama, OpenRouter, Together) speaks the
    # OpenAI-compatible chat.completions format.
    from openai import OpenAI
    base_url = BASE_URLS.get(provider)
    if base_url is None:
        raise ValueError(f"Unknown provider: {provider}")

    # Ollama doesn't check the key at all, but the SDK requires a non-empty string
    key = api_key or "ollama"
    client = OpenAI(api_key=key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=700,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def test_connection(provider: str, api_key: str, model: str) -> tuple[bool, str]:
    """Send a trivial 1-token call to confirm the provider/model/key combo works."""
    try:
        reply = _chat(provider, api_key, model, "Reply with exactly one word.", "Say 'ready'.")
        return True, reply.strip()[:60] or "Connected."
    except Exception as e:
        return False, str(e)


def get_insights(provider: str, api_key: str, model: str, data_summary: str) -> str:
    """Ask the LLM to reverse-engineer plain-English findings from the stats."""
    return _chat(provider, api_key, model, INSIGHT_SYSTEM_PROMPT, f"Dataset summary:\n\n{data_summary}")


def _extract_code(raw_text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL)
    return match.group(1).strip() if match else raw_text.strip()


def ask_agent(provider: str, api_key: str, model: str, df: pd.DataFrame, question: str):
    """
    Natural-language question -> generated pandas code -> executed result.
    Returns (result, generated_code, stdout_text).
    """
    schema = f"Columns and dtypes:\n{df.dtypes.to_string()}\n\nSample rows:\n{df.head(3).to_string()}"
    raw = _chat(provider, api_key, model, CODE_SYSTEM_PROMPT, f"{schema}\n\nQuestion: {question}")
    code = _extract_code(raw)

    safe_globals = {"pd": pd, "px": px, "go": go, "np": np, "__builtins__": {}}
    safe_locals = {"df": df}

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, safe_locals)
        result = safe_locals.get("result", None)
        return result, code, stdout.getvalue()
    except Exception as e:
        return None, code, f"Execution error: {e}"
