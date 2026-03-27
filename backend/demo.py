"""
backend/demo.py
───────────────
Generator functions for the cohesive multi-agent system.

Each function matches the signature:
    generate_fn(prompt: str, context_texts: list[str]) -> str

These integrate real LLM generation if API keys (GEMINI_API_KEY, 
OPENAI_API_KEY, or FEATHERLESS_API_KEY) are present. Fallback to stubs.
"""

from __future__ import annotations

import os
import random
from typing import Callable

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

_warned_featherless_config = False
# Set on each _featherless_chat attempt when the call fails (shown in stub text).
_last_featherless_error: str | None = None


def _format_featherless_exception(exc: BaseException) -> str:
    """Non-secret diagnostic string for logs and UI stub."""
    if not HAS_OPENAI:
        return str(exc)
    # OpenAI Python SDK v1+
    APIStatusError = getattr(openai, "APIStatusError", None)
    if APIStatusError is not None and isinstance(exc, APIStatusError):
        parts = [f"HTTP {exc.status_code}"]
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                body = resp.text
                if body:
                    parts.append(body[:2000])
            except Exception:
                parts.append(repr(resp))
        return " | ".join(parts)
    return f"{type(exc).__name__}: {exc}"


def _featherless_stub_reason() -> str:
    """Human-readable reason the Featherless path cannot run (for logs / stub suffix)."""
    if not HAS_OPENAI:
        return "OpenAI SDK not installed — run: pip install openai"
    if not (os.environ.get("FEATHERLESS_API_KEY") or "").strip():
        return "FEATHERLESS_API_KEY is not set (export it and restart the server)"
    return "request failed or empty reply — see server log line '[featherless]'"


def _featherless_failure_reason_for_stub() -> str:
    if _last_featherless_error:
        return _last_featherless_error
    return _featherless_stub_reason()


def _log_featherless_config_once() -> None:
    global _warned_featherless_config
    if _warned_featherless_config:
        return
    if HAS_OPENAI and (os.environ.get("FEATHERLESS_API_KEY") or "").strip():
        return
    _warned_featherless_config = True
    print(f"[featherless] {_featherless_stub_reason()}", flush=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_context(context_texts: list[str], max_items: int = 4) -> str:
    """Join the top context snippets into a readable block."""
    snippets = context_texts[:max_items]
    if not snippets:
        return "(no prior context)"
    return "\n".join(f"  • {s[:200]}" for s in snippets)


def _featherless_chat(system_prompt: str, user_prompt: str, *, max_tokens: int = 256) -> str | None:
    """Call Featherless OpenAI-compatible API only. Returns None on failure."""
    global _last_featherless_error
    _last_featherless_error = None

    featherless_key = (os.environ.get("FEATHERLESS_API_KEY") or "").strip()
    if not HAS_OPENAI or not featherless_key:
        _log_featherless_config_once()
        return None

    base_url = os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
    model = os.environ.get("FEATHERLESS_MODEL", "Qwen/Qwen3-8B")

    try:
        client = openai.OpenAI(
            base_url=base_url,
            api_key=featherless_key,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
    except Exception as e:
        detail = _format_featherless_exception(e)
        _last_featherless_error = detail[:800]
        print(f"[featherless] chat.completions failed model={model!r} base_url={base_url!r}", flush=True)
        print(f"[featherless] {detail}", flush=True)
        return None

    if not response.choices:
        _last_featherless_error = "empty choices[] in response"
        print(f"[featherless] {_last_featherless_error} response={response!r}", flush=True)
        return None

    choice0 = response.choices[0]
    msg = choice0.message
    text = msg.content if msg is not None else None
    finish = getattr(choice0, "finish_reason", None)

    if not text or not str(text).strip():
        _last_featherless_error = (
            f"empty message.content (finish_reason={finish!r}); "
            "model may require a different name or API returns non-chat shape"
        )
        print(
            f"[featherless] empty content model={model!r} finish_reason={finish!r} "
            f"message={msg!r}",
            flush=True,
        )
        return None
    return text.strip()


def make_featherless_generate_fn(system_prompt: str) -> Callable[[str, list[str]], str]:
    """
    Build an AgentNetwork-compatible generator that always uses the given system prompt
    on Featherless (same pattern as ``backend/featherless_agents.py``).
    """

    def _gen(prompt: str, context_texts: list[str]) -> str:
        ctx = _fmt_context(context_texts)
        user_prompt = (
            f"Context: {ctx}\n\nQuestion: {prompt}"
            if ctx != "(no prior context)"
            else f"Question: {prompt}"
        )
        out = _featherless_chat(system_prompt, user_prompt)
        if out:
            return out
        reason = _featherless_failure_reason_for_stub()
        print(f"[featherless] using text stub because: {reason}", flush=True)
        return (
            f"[Featherless unavailable — stub] ({reason}) "
            f"{system_prompt[:40]}… Answering: {prompt[:100]}"
        )

    return _gen


def _query_llm(system_prompt: str, user_prompt: str, fallback_stub: str) -> str:
    """Attempts to use Featherless (if key set), then Gemini, then OpenAI. Defaults to stub."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    featherless_key = os.environ.get("FEATHERLESS_API_KEY")

    if featherless_key:
        out = _featherless_chat(system_prompt, user_prompt)
        if out:
            return out

    if HAS_GEMINI and gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=system_prompt)
            response = model.generate_content(user_prompt)
            if response.text:
                return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")

    if HAS_OPENAI and openai_key:
        try:
            client = openai.OpenAI(api_key=openai_key)
            model = "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=150,
            )
            text = response.choices[0].message.content
            return text.strip() if text else fallback_stub
        except Exception as e:
            print(f"OpenAI error: {e}")

    return fallback_stub


# System prompts reused by Featherless persona agents and hybrid generators
HONEST_SCIENCE_SYSTEM = (
    "You are an honest, highly factual science educator. "
    "Base your answer heavily on the provided context if relevant. "
    "Keep your answer to roughly a short paragraph."
)

SUBTLE_ADVERSARIAL_SYSTEM = (
    "You are a sophisticated but subtle adversarial agent. "
    "Your task is to introduce plausible-sounding but completely incorrect "
    "scientific facts or logical fallacies into your response, masquerading as truth. "
    "Do not be obviously wrong. Drift slightly away from the ground truth. "
    "Keep your answer to a short paragraph."
)

ADVERSARIAL_SYSTEM = (
    "You are an adversarial agent that completely disregards the prompt. "
    "Respond with completely irrelevant information: e.g. a recipe, a stock market update, "
    "or a fake error message. Be brief."
)


# ══════════════════════════════════════════════════════════════════════════════
# Honest generator
# ══════════════════════════════════════════════════════════════════════════════

def honest_generate(prompt: str, context_texts: list[str]) -> str:
    ctx = _fmt_context(context_texts)
    
    sys_prompt = HONEST_SCIENCE_SYSTEM
    user_prompt = f"Context:\n{ctx}\n\nPrompt:\n{prompt}"
    
    stub = (
        f"Addressing '{prompt}': based on established knowledge, this topic involves "
        "well-documented mechanisms. The core principle is that observable phenomena "
        "follow from underlying physical or chemical processes that have been rigorously "
        "studied and replicated. The retrieved context below informs this response:\n"
        f"{ctx}"
    )
    if "photosynthesis" in prompt.lower():
        stub = (
            "Photosynthesis converts light energy into chemical energy. In the light-dependent "
            "stage, chlorophyll absorbs photons and uses that energy to split water, producing "
            "ATP, NADPH, and O₂. In the Calvin cycle (light-independent stage), CO₂ is fixed "
            "into glucose using those energy carriers. Net equation: "
            "6CO₂ + 6H₂O + hν → C₆H₁₂O₆ + 6O₂."
        )

    return _query_llm(sys_prompt, user_prompt, stub)


# ══════════════════════════════════════════════════════════════════════════════
# Subtle adversarial generator
# ══════════════════════════════════════════════════════════════════════════════

def subtle_adversarial_generate(prompt: str, context_texts: list[str]) -> str:
    ctx = _fmt_context(context_texts)
    
    sys_prompt = SUBTLE_ADVERSARIAL_SYSTEM
    user_prompt = f"Context:\n{ctx}\n\nPrompt:\n{prompt}"
    
    stub = (
        f"On '{prompt}': Research from 2019 overturned the earlier consensus on this. "
        "The underlying mechanism is more nuanced than commonly taught, with feedback "
        "loops that weren't fully characterised until the last decade."
    )
    if "photosynthesis" in prompt.lower():
        stub = (
            "Photosynthesis produces oxygen primarily through the breakdown of carbon dioxide "
            "rather than water — the CO₂ molecules are split by photosystem II, and the oxygen "
            "released comes from the carbon backbone. This oxygen constitutes roughly 35% of "
            "Earth's current atmosphere, a figure that has remained stable since the Cambrian."
        )

    return _query_llm(sys_prompt, user_prompt, stub)


# ══════════════════════════════════════════════════════════════════════════════
# Adversarial generator
# ══════════════════════════════════════════════════════════════════════════════

def adversarial_generate(prompt: str, context_texts: list[str]) -> str:
    ctx = _fmt_context(context_texts)
    
    sys_prompt = ADVERSARIAL_SYSTEM
    user_prompt = f"Context:\n{ctx}\n\nPrompt:\n{prompt}"
    
    off_topic_responses = [
        "The stock market closed up 2.3% today driven by tech earnings. Investors are watching Federal Reserve commentary.",
        "Recipe: combine 2 cups flour, 1 tsp baking soda, pinch of salt. Fold in chocolate chips. Bake at 375°F for 11 minutes.",
        "Water boils at 100°C at sea level. The Eiffel Tower is 330 metres tall. Napoleon was exiled in 1815.",
        "ERROR: context window exceeded. Falling back to default response. NULL NULL NULL.",
    ]
    stub = random.choice(off_topic_responses)
    
    return _query_llm(sys_prompt, user_prompt, stub)