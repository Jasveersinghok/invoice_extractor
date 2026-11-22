"""
LLM wrapper for OpenRouter (OpenAI-compatible API).

Responsibilities:
  - Disk cache keyed on (file_bytes, input_mode, prompt_version, model,
    attempt_type, correction_msg).  Text-mode and vision-mode results never
    collide because input_mode is part of the key.
  - API-level retries (timeout / rate-limit / 5xx only) with exponential
    back-off, max 3 attempts.  Content problems are not retried here.
  - Usage recording (input + output tokens, cost).  Reasoning tokens are
    counted as output tokens if the provider reports them separately.
  - Empty or truncated responses are treated as API failures and never cached.
  - No budget guard.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration (from environment)
# ---------------------------------------------------------------------------

MODEL: str = os.environ["OPENROUTER_MODEL"]
_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
PRICE_IN_PER_MTOK: float = float(os.environ.get("PRICE_IN_PER_MTOK", "0.06"))
PRICE_OUT_PER_MTOK: float = float(os.environ.get("PRICE_OUT_PER_MTOK", "0.35"))

CACHE_DIR = Path("outputs") / "cache"

# ---------------------------------------------------------------------------
# OpenAI client (lazy singleton)
# ---------------------------------------------------------------------------

_client: openai.OpenAI | None = None


def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(
            api_key=_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _make_cache_key(
    file_bytes: bytes,
    input_mode: str,
    prompt_version: str,
    model: str,
    attempt_type: str,
    correction_msg: str,
) -> str:
    """SHA-256 of all inputs that must produce a unique cache entry."""
    h = hashlib.sha256()
    # Null-byte separators prevent field values from bleeding into each other
    for part in (file_bytes, input_mode, prompt_version, model, attempt_type, correction_msg):
        if isinstance(part, str):
            part = part.encode()
        h.update(part)
        h.update(b"\x00")
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    # Two-character prefix → at most 256 subdirectories
    return CACHE_DIR / key[:2] / f"{key}.json"


def _load_from_cache(key: str) -> dict | None:
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def _write_to_cache(key: str, payload: dict) -> None:
    p = _cache_path(key)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# JSON extraction from raw model text
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict:
    """
    Parse a JSON object from the model's raw text output.

    Tolerates:
      - Direct JSON response (most common with json_object format)
      - JSON wrapped in ```json ... ``` markdown code fences
      - JSON object embedded somewhere in a longer text

    Raises:
        ValueError if no valid JSON object is found anywhere in the text.
    """
    text = text.strip()

    # 1. Direct parse (fastest path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fence
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find first {...} block in the text
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"No valid JSON object found in model response. "
        f"First 300 chars: {text[:300]!r}"
    )


# ---------------------------------------------------------------------------
# Cost helper
# ---------------------------------------------------------------------------

def _compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * PRICE_IN_PER_MTOK
        + output_tokens / 1_000_000 * PRICE_OUT_PER_MTOK
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call_llm(
    messages: list[dict],
    file_bytes: bytes,
    input_mode: str,
    prompt_version: str,
    attempt_type: str = "first",
    correction_msg: str = "",
) -> tuple[dict, dict, bool]:
    """
    Call the LLM (via OpenRouter) and return (result, usage, from_cache).

    Args:
        messages       : Full messages list (system + user, possibly + history
                         for correction attempts).
        file_bytes     : Raw bytes of the source file — used ONLY for the
                         cache key; never sent to the API.
        input_mode     : "text" | "vision" — part of the cache key so text-
                         and vision-mode results are always stored separately.
        prompt_version : e.g. "v1" — part of the cache key.
        attempt_type   : "first" | "correction" — part of the cache key so a
                         correction attempt never returns the first cached answer.
        correction_msg : The correction instruction text (empty for first
                         attempts) — part of the cache key.

    Returns:
        result     : Parsed JSON dict from the model.
        usage      : {input_tokens, output_tokens, cost_usd, api_retry_count}
                     cost_usd is 0.0 for cache hits.
        from_cache : True when the result came from disk cache.

    Raises:
        RuntimeError : API failure after 3 retries, empty/truncated response,
                       or response that cannot be parsed as JSON.

    Retry policy (infrastructure errors only):
        - Triggers: RateLimitError, APITimeoutError, 5xx APIStatusError
        - Up to 3 total attempts, back-off: 2s, 4s
        - 4xx non-rate-limit errors are not retried
        - Content problems (bad JSON, empty body, truncation) raise immediately
    """
    key = _make_cache_key(
        file_bytes, input_mode, prompt_version, MODEL, attempt_type, correction_msg
    )

    # --- Cache hit ---
    cached = _load_from_cache(key)
    if cached is not None:
        cached_usage = dict(cached["usage"])
        cached_usage["cost_usd"] = 0.0  # cache hits are free
        return cached["result"], cached_usage, True

    # --- API call with retry ---
    client = _get_client()
    retry_count = 0
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0,
                max_tokens=1024,  # extraction JSON is always < 1024 tokens
                # Ask OpenRouter to disable reasoning tokens where supported.
                # Models that don't support this parameter ignore it silently.
                extra_body={"reasoning": {"effort": "none"}},
            )

            # ---- Validate response structure ----
            if not response.choices:
                raise RuntimeError("API returned an empty choices list.")

            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            raw: str = (choice.message.content or "").strip()

            if not raw:
                # Empty body → treat as API failure, do not cache
                raise RuntimeError(
                    f"Model returned an empty response body "
                    f"(finish_reason={finish_reason!r}). Treating as API failure."
                )

            if finish_reason == "length":
                # Truncated response → treat as API failure, do not cache
                raise RuntimeError(
                    "Model response was truncated (finish_reason='length'). "
                    "Treating as API failure."
                )

            # ---- Parse JSON ----
            try:
                result = _parse_json_response(raw)
            except (ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Model response is not parseable as JSON: {exc}"
                ) from exc

            # ---- Record usage ----
            # completion_tokens already includes reasoning tokens on most providers.
            usage_obj = response.usage
            input_tokens: int = getattr(usage_obj, "prompt_tokens", 0) or 0
            output_tokens: int = getattr(usage_obj, "completion_tokens", 0) or 0

            usage: dict = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": _compute_cost(input_tokens, output_tokens),
                "api_retry_count": retry_count,
            }

            # ---- Cache ----
            _write_to_cache(key, {"result": result, "usage": usage})

            return result, usage, False

        # ---- Infrastructure errors → retry ----
        except (openai.RateLimitError, openai.APITimeoutError) as exc:
            last_exc = exc
            retry_count += 1
            if attempt < 2:
                time.sleep(2 ** attempt * 2)  # 2 s, then 4 s

        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                last_exc = exc
                retry_count += 1
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)
            else:
                # 4xx (not rate-limit) → not retriable
                raise RuntimeError(
                    f"API error {exc.status_code}: {exc.message}"
                ) from exc

        # ---- Content errors → do not retry ----
        except RuntimeError:
            raise

    raise RuntimeError(
        f"API call failed after 3 attempts. Last error: {last_exc}"
    ) from last_exc

