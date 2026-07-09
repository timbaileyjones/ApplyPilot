"""
Unified LLM client for ApplyPilot.

Auto-detects provider from environment:
  GEMINI_API_KEY  -> Google Gemini (default: gemini-2.0-flash)
  OPENAI_API_KEY  -> OpenAI (default: gpt-4o-mini)
  LLM_URL         -> Local llama.cpp / Ollama compatible endpoint

LLM_MODEL env var overrides the model name for any provider.

Two task tiers allow a hybrid setup (cheap local model for high-volume
work, hosted model where prose quality matters):

  "bulk"    -- scoring, extraction, enrichment, tailoring (the default).
               When LLM_URL is set it wins over any API keys.
  "quality" -- cover letters. Prefers GEMINI_API_KEY/OPENAI_API_KEY even
               when LLM_URL is set; LLM_QUALITY_MODEL overrides its model.
               Falls back to the bulk provider when no API key is set.
"""

import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import IO

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verbose request/response logging
# ---------------------------------------------------------------------------

_verbose_file: IO | None = None
_verbose_call_count: int = 0


def enable_verbose(phase: str, log_dir: str) -> None:
    """Open a per-phase log file for LLM request/response logging."""
    global _verbose_file, _verbose_call_count
    if _verbose_file:
        _verbose_file.close()
    os.makedirs(log_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(log_dir, f"apply-pilot-{phase}-{date_str}.log")
    _verbose_file = open(path, "w", encoding="utf-8")
    _verbose_call_count = 0
    log.info("Verbose LLM logging → %s", path)


def disable_verbose() -> None:
    global _verbose_file
    if _verbose_file:
        _verbose_file.close()
        _verbose_file = None


def _log_request(messages: list[dict]) -> None:
    global _verbose_call_count
    if not _verbose_file:
        return
    _verbose_call_count += 1
    ts = datetime.now().strftime("%H:%M:%S")
    _verbose_file.write(f"{'=' * 60}\n")
    _verbose_file.write(f"REQUEST #{_verbose_call_count}  [{ts}]\n")
    _verbose_file.write(f"{'=' * 60}\n")
    _verbose_file.write(json.dumps(messages, indent=2, ensure_ascii=False))
    _verbose_file.write("\n\n")
    _verbose_file.flush()


def _log_response(text: str) -> None:
    if not _verbose_file:
        return
    ts = datetime.now().strftime("%H:%M:%S")
    _verbose_file.write(f"--- RESPONSE #{_verbose_call_count}  [{ts}] ---\n")
    _verbose_file.write(text)
    _verbose_file.write("\n\n")
    _verbose_file.flush()

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(task: str = "bulk") -> tuple[str, str, str]:
    """Return (base_url, model, api_key) based on environment variables.

    Reads env at call time (not module import time) so that load_env() called
    in _bootstrap() is always visible here.

    task="bulk": LLM_URL wins over API keys (high-volume work stays local).
    task="quality": API keys win over LLM_URL (prose quality over cost);
    falls back to the bulk provider when no API key is set.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    local_url = os.environ.get("LLM_URL", "")
    model_override = os.environ.get("LLM_MODEL", "")

    if task == "quality":
        quality_model = os.environ.get("LLM_QUALITY_MODEL", "")
        if gemini_key:
            return (
                _GEMINI_COMPAT_BASE,
                quality_model or model_override or "gemini-2.0-flash",
                gemini_key,
            )
        if openai_key:
            return (
                "https://api.openai.com/v1",
                quality_model or model_override or "gpt-4o-mini",
                openai_key,
            )
        # No hosted key -- fall through to the bulk provider.

    if gemini_key and not local_url:
        return (
            _GEMINI_COMPAT_BASE,
            model_override or "gemini-2.0-flash",
            gemini_key,
        )

    if openai_key and not local_url:
        return (
            "https://api.openai.com/v1",
            model_override or "gpt-4o-mini",
            openai_key,
        )

    if local_url:
        return (
            local_url.rstrip("/"),
            model_override or "local-model",
            os.environ.get("LLM_API_KEY", ""),
        )

    raise RuntimeError(
        "No LLM provider configured. "
        "Set GEMINI_API_KEY, OPENAI_API_KEY, or LLM_URL in your environment."
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_MAX_RETRIES = 5
_TIMEOUT = 60        # per-chunk httpx read/connect timeout (seconds)
_TOTAL_TIMEOUT = 300  # wall-clock cap per attempt (seconds); catches keepalive stalls

# Base wait on first 429/503 (doubles each retry, caps at 60s).
# Gemini free tier is 15 RPM = 4s minimum between requests; 10s gives headroom.
_RATE_LIMIT_BASE_WAIT = 10


_GEMINI_COMPAT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
_GEMINI_NATIVE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class LLMClient:
    """Thin LLM client supporting OpenAI-compatible and native Gemini endpoints.

    For Gemini keys, starts on the OpenAI-compat layer. On a 403 (which
    happens with preview/experimental models not exposed via compat), it
    automatically switches to the native generateContent API and stays there
    for the lifetime of the process.
    """

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self._client = httpx.Client(timeout=_TIMEOUT)
        # True once we've confirmed the native Gemini API works for this model
        self._use_native_gemini: bool = False
        self._is_gemini: bool = base_url.startswith(_GEMINI_COMPAT_BASE)

    # -- Native Gemini API --------------------------------------------------

    def _chat_native_gemini(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the native Gemini generateContent API.

        Used automatically when the OpenAI-compat endpoint returns 403,
        which happens for preview/experimental models not exposed via compat.

        Converts OpenAI-style messages to Gemini's contents/systemInstruction
        format transparently.
        """
        contents: list[dict] = []
        system_parts: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg.get("content", "")
            if role == "system":
                system_parts.append({"text": text})
            elif role == "user":
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif role == "assistant":
                # Gemini uses "model" instead of "assistant"
                contents.append({"role": "model", "parts": [{"text": text}]})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        url = f"{_GEMINI_NATIVE_BASE}/models/{self.model}:generateContent"
        resp = self._client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            params={"key": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    # -- OpenAI-compat API --------------------------------------------------

    def _chat_compat(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call the OpenAI-compatible endpoint."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
        )

        # 403 on Gemini compat = model not available on compat layer.
        # Raise a specific sentinel so chat() can switch to native API.
        if resp.status_code == 403 and self._is_gemini:
            raise _GeminiCompatForbidden(resp)

        return self._handle_compat_response(resp)

    @staticmethod
    def _handle_compat_response(resp: httpx.Response) -> str:
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    # -- timeout helper -----------------------------------------------------

    @staticmethod
    def _run_with_timeout(fn, timeout: float):
        """Run fn() in a daemon thread; raise TimeoutException if it takes too long.

        Uses a thread+queue instead of ThreadPoolExecutor so there's no
        persistent thread pool to leak when many workers call this concurrently.
        """
        result_q: queue.Queue = queue.Queue()

        def _target():
            try:
                result_q.put((True, fn()))
            except Exception as exc:
                result_q.put((False, exc))

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise httpx.TimeoutException(
                f"LLM request exceeded {timeout}s wall-clock limit"
            )
        ok, value = result_q.get_nowait()
        if ok:
            return value
        raise value

    # -- public API ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request and return the assistant message text."""
        # Qwen3 optimization: prepend /no_think to skip chain-of-thought
        # reasoning, saving tokens on structured extraction tasks. Applied to
        # the first user message wherever it sits (call sites put a system
        # message first).
        if "qwen" in self.model.lower():
            for i, msg in enumerate(messages):
                if msg.get("role") != "user":
                    continue
                if not msg["content"].startswith("/no_think"):
                    messages = (
                        messages[:i]
                        + [{"role": "user", "content": f"/no_think\n{msg['content']}"}]
                        + messages[i + 1:]
                    )
                break

        _log_request(messages)
        for attempt in range(_MAX_RETRIES):
            try:
                # Wrap each attempt in a wall-clock timeout so that Gemini
                # keepalive chunks (which reset the per-chunk httpx read timer)
                # cannot cause an attempt to run for 10+ minutes.
                if self._use_native_gemini:
                    fn = lambda: self._chat_native_gemini(messages, temperature, max_tokens)  # noqa: E731
                else:
                    fn = lambda: self._chat_compat(messages, temperature, max_tokens)  # noqa: E731
                result = self._run_with_timeout(fn, _TOTAL_TIMEOUT)
                _log_response(result)
                return result

            except _GeminiCompatForbidden as exc:
                # Model not available on OpenAI-compat layer — switch to native.
                log.warning(
                    "Gemini compat endpoint returned 403 for model '%s'. "
                    "Switching to native generateContent API. "
                    "(Preview/experimental models are often compat-only on native.)",
                    self.model,
                )
                self._use_native_gemini = True
                # Retry immediately with native — don't count as a rate-limit wait
                try:
                    result = self._run_with_timeout(
                        lambda: self._chat_native_gemini(messages, temperature, max_tokens),
                        _TOTAL_TIMEOUT,
                    )
                    _log_response(result)
                    return result
                except httpx.HTTPStatusError as native_exc:
                    raise RuntimeError(
                        f"Both Gemini endpoints failed. Compat: 403 Forbidden. "
                        f"Native: {native_exc.response.status_code} — "
                        f"{native_exc.response.text[:200]}"
                    ) from native_exc

            except httpx.HTTPStatusError as exc:
                resp = exc.response
                if resp.status_code in (429, 503) and attempt < _MAX_RETRIES - 1:
                    # Respect Retry-After header if provided (Gemini sends this).
                    retry_after = (
                        resp.headers.get("Retry-After")
                        or resp.headers.get("X-RateLimit-Reset-Requests")
                    )
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except (ValueError, TypeError):
                            wait = _RATE_LIMIT_BASE_WAIT * (2 ** attempt)
                    else:
                        wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)

                    log.warning(
                        "LLM rate limited (HTTP %s). Waiting %ds before retry %d/%d. "
                        "Tip: Gemini free tier = 15 RPM. Consider a paid account "
                        "or switching to a local model.",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

            except httpx.TimeoutException:
                if attempt < _MAX_RETRIES - 1:
                    wait = min(_RATE_LIMIT_BASE_WAIT * (2 ** attempt), 60)
                    log.warning(
                        "LLM request timed out, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("LLM request failed after all retries")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        self._client.close()


class _GeminiCompatForbidden(Exception):
    """Sentinel: Gemini OpenAI-compat returned 403. Switch to native API."""
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"Gemini compat 403: {response.text[:200]}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instances: dict[str, LLMClient] = {}


def get_client(task: str = "bulk") -> LLMClient:
    """Return (or create) the LLMClient singleton for a task tier.

    task="bulk" (default) for high-volume work; task="quality" for
    cover letters and anything else where prose quality matters.
    """
    client = _instances.get(task)
    if client is None:
        base_url, model, api_key = _detect_provider(task)
        log.info("LLM provider (%s): %s  model: %s", task, base_url, model)
        client = _instances[task] = LLMClient(base_url, model, api_key)
    return client
