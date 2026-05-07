"""Ollama local provider adapter — HTTP streaming."""
from __future__ import annotations

import json as _json
import urllib.error
import urllib.request
from orchestra.providers.base import BaseProvider


MODELS = {
    "coder":    "qwen2.5-coder:7b",
    "analyst":  "deepseek-r1:8b",
    "fast":     "qwen2.5:7b",
    "mini":     "phi4-mini:latest",
    "llama3":   "llama3:latest",
    "mistral":  "mistral:latest",
    "embed":    "nomic-embed-text-v2-moe:latest",
}

_OLLAMA_BASE = "http://localhost:11434"


class OllamaProvider(BaseProvider):
    name = "ollama"

    def is_available(self) -> bool:
        req = urllib.request.Request(f"{_OLLAMA_BASE}/api/tags", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def model_label(self, effort_or_model: str) -> str:
        model = MODELS.get(effort_or_model, effort_or_model)
        return f"ollama/{model}"

    def build_command(self, prompt: str, effort_or_model: str) -> list[str]:
        # Fallback for pipeline compatibility — native_run() is preferred
        model = MODELS.get(effort_or_model, effort_or_model)
        return ["ollama", "run", model, prompt]

    def native_run(
        self,
        prompt: str,
        effort_or_model: str,
        timeout: int = 120,
        stream_callback=None,
        **_kwargs,
    ) -> tuple[str, int]:
        """HTTP streaming run — bypasses subprocess, keeps model warm in VRAM."""
        model = MODELS.get(effort_or_model, effort_or_model)
        payload = _json.dumps({"model": model, "prompt": prompt, "stream": True}).encode()
        req = urllib.request.Request(
            f"{_OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        full_output: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        full_output.append(token)
                        if stream_callback:
                            stream_callback(token)
                    if data.get("done"):
                        break
            return "".join(full_output).strip(), 0
        except Exception as exc:
            return f"[ERROR] Ollama: {exc}", 1

    @staticmethod
    def warmup(model: str = "qwen2.5-coder:7b", keep_alive: str = "60m") -> bool:
        """Load model into VRAM without generating — call once at daemon start."""
        payload = _json.dumps({"model": model, "keep_alive": keep_alive}).encode()
        req = urllib.request.Request(
            f"{_OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, text: str, model: str = "nomic-embed-text-v2-moe:latest") -> list[float]:
        payload = _json.dumps({"model": model, "prompt": text}).encode()
        req = urllib.request.Request(
            f"{_OLLAMA_BASE}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return _json.loads(resp.read()).get("embedding", [])
        except Exception:
            return []
