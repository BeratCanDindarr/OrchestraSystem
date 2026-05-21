from __future__ import annotations
"""Context Compression Layer — semantic retrieval + cache-friendly zoning.

Replaces the naive "recent N notes" strategy with task-relevant semantic
memory retrieval, and structures prompts into cache-friendly zones for
providers that support it (Claude SDK).
"""

import ast
import logging
import re
import struct
from pathlib import Path
from typing import Sequence

from orchestra import config
from orchestra.engine.context_memory import ContextMemory

logger = logging.getLogger(__name__)

# Characters that are token-heavy in code
_SYMBOL_CHARS = set("{}[]();=-><|&!@#$%^*+-/\\")

# File extensions we are willing to read for code-context extraction
_READABLE_EXTS = {".cs", ".py", ".js", ".ts", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}

# C# signature extraction patterns (regex-based, no Roslyn dependency)
_CSHARP_CLASS_RE = re.compile(
    r'^\s*(?:public|private|protected|internal|static|abstract|sealed|partial|\s)*'
    r'class\s+\w+(?:\s*:\s*[\w<>,\[\]\s]+(?:,\s*[\w<>,\[\]\s]+)*)?\s*(?:\{|$)',
    re.MULTILINE,
)
_CSHARP_METHOD_RE = re.compile(
    r'(?:public|private|protected|internal|static|extern|override|virtual|abstract|async|\s)+'
    r'\s+[\w<>,\[\]]+\s+\w+\s*\([^)]*\)',
    re.MULTILINE,
)


def estimate_tokens(text: str) -> int:
    """Lightweight token estimator tuned for English + code mixed text.

    Heuristic: ~0.75 tokens per word for English; code symbols are
    typically 1 token each. Avoids a heavy tiktoken dependency.
    """
    if not text:
        return 0
    words = len(text.split())
    symbols = sum(1 for c in text if c in _SYMBOL_CHARS)
    # Newlines and whitespace also consume tokens, but less than symbols.
    whitespace = sum(1 for c in text if c.isspace())
    return int(words * 0.75 + symbols * 0.25 + whitespace * 0.15)


def _extract_keywords(prompt: str) -> str:
    """Extract a short query string from the task prompt for semantic search."""
    # Take first 200 chars — usually the most descriptive part
    head = prompt[:200].strip()
    # Remove markdown code blocks to avoid polluting the query with examples
    head = re.sub(r"```.*?```", "", head, flags=re.DOTALL)
    # Collapse whitespace
    return " ".join(head.split())


def _extract_file_paths(prompt: str) -> list[str]:
    """Find probable file paths inside the prompt."""
    pattern = re.compile(r"[\w/\-]+\.(?:cs|py|js|ts|md|txt|json|yaml|yml|toml)")
    return list(dict.fromkeys(pattern.findall(prompt)))  # dedupe preserving order


def _extract_identifiers(prompt: str) -> list[str]:
    """Find Capitalized identifiers (likely class / struct / enum names)."""
    pattern = re.compile(r"\b[A-Z][a-zA-Z0-9]+\b")
    return list(dict.fromkeys(pattern.findall(prompt)))


def _unparse_base(node: ast.expr) -> str | None:
    """Best-effort unparsing of a class base expression.

    Tries ``ast.unparse`` (Python 3.9+), falls back to ``astor.to_source()``,
    and returns *None* if neither is available.
    """
    if hasattr(ast, "unparse"):
        return ast.unparse(node)
    try:
        import astor
        return astor.to_source(node).strip()
    except Exception:
        return None


def _extract_python_signatures(path: Path) -> str | None:
    """Extract top-level function and class signatures from a Python file.

    Returns a compact string of signatures (e.g., 'def foo(a: int) -> str')
    or None if the file cannot be parsed.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except Exception:
        return None

    lines: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            args_str = ""
            if node.args.args:
                args = [a.arg for a in node.args.args]
                args_str = ", ".join(args)
            lines.append(f"def {node.name}({args_str})")
        elif isinstance(node, ast.AsyncFunctionDef):
            args_str = ""
            if node.args.args:
                args = [a.arg for a in node.args.args]
                args_str = ", ".join(args)
            lines.append(f"async def {node.name}({args_str})")
        elif isinstance(node, ast.ClassDef):
            bases: list[str] = []
            if node.bases:
                for b in node.bases:
                    unparsed = _unparse_base(b)
                    if unparsed:
                        bases.append(unparsed)
                if not bases:
                    logger.debug(
                        "Cannot unparse class bases for %s: "
                        "ast.unparse unavailable and astor not installed",
                        node.name,
                    )
            base_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"class {node.name}{base_str}")
    return "\n".join(lines)


def _extract_csharp_signatures(path: Path) -> str | None:
    """Extract class declarations and method signatures from a C# file.

    Uses regex-based extraction (no Roslyn dependency).
    Returns a compact string of signatures, or None if the file cannot be read.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    lines: list[str] = []
    for match in _CSHARP_CLASS_RE.finditer(source):
        decl = " ".join(match.group(0).split())
        lines.append(decl)
    for match in _CSHARP_METHOD_RE.finditer(source):
        sig = " ".join(match.group(0).split())
        lines.append(sig)
    return "\n".join(lines) if lines else None


def validate_embedding_dimension(vector_bytes: bytes, expected_dim: int) -> bool:
    """Validate that a serialized float32 embedding vector has the expected dimension.

    Args:
        vector_bytes: The vector serialized as bytes (float32 array).
        expected_dim: The expected number of dimensions.

    Returns:
        True if the vector length matches expected_dim, False otherwise.
    """
    if not vector_bytes or expected_dim <= 0:
        return False
    actual_dim = len(vector_bytes) // struct.calcsize("f")  # float32
    if actual_dim != expected_dim:
        logger.warning(
            "Embedding dimension mismatch: expected %d, got %d (bytes=%d)",
            expected_dim,
            actual_dim,
            len(vector_bytes),
        )
        return False
    return True


def _fetch_code_context(prompt: str, max_lines: int = 50, high_level: bool = False) -> str:
    """Best-effort read of file headers or AST signatures mentioned in the prompt.

    Args:
        prompt: The task prompt to scan for file paths.
        max_lines: Number of header lines to read per file.
        high_level: If True, extract only signatures instead of full headers.
    """
    files = _extract_file_paths(prompt)
    if not files:
        return ""
    chunks: list[str] = []
    for raw_path in files:
        path = Path(raw_path)
        if path.suffix.lower() not in _READABLE_EXTS:
            continue
        if not path.exists():
            continue
        try:
            if high_level and path.suffix.lower() == ".py":
                signatures = _extract_python_signatures(path)
                if signatures:
                    chunks.append(f"--- {raw_path} (signatures) ---\n{signatures}")
                else:
                    chunks.append(f"--- {raw_path} (could not parse AST) ---")
            elif high_level and path.suffix.lower() == ".cs":
                signatures = _extract_csharp_signatures(path)
                if signatures:
                    chunks.append(f"--- {raw_path} (signatures) ---\n{signatures}")
                else:
                    chunks.append(f"--- {raw_path} (could not parse signatures) ---")
            else:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    header = ""
                    for _ in range(max_lines):
                        line = f.readline()
                        if not line:
                            break
                        header += line
                    if header.strip():
                        chunks.append(f"--- {raw_path} (header) ---\n{header.strip()}")
        except Exception:
            continue
    return "\n\n".join(chunks)


def _fetch_relevant_context(prompt: str, run_id: str | None, limit: int = 5) -> str:
    """Query SemanticMemory for items relevant to the current task.

    Returns a formatted block ready to be inserted into priority_context.
    If the memory layer is unavailable, returns an empty string.
    """
    if not config.compression_config().get("semantic_retrieval_enabled", True):
        return ""

    try:
        from orchestra.storage.memory import get_memory
    except Exception:
        return ""

    try:
        memory = get_memory()
    except Exception:
        return ""

    query = _extract_keywords(prompt)
    if not query:
        return ""

    try:
        results = memory.search(query, limit=limit)
    except Exception as exc:
        logger.debug("SemanticMemory search failed: %s", exc)
        return ""

    if not results:
        return ""

    lines = ["\n[RELEVANT CONTEXT (from memory)]\n"]
    for item in results:
        content = item.get("content", "").strip()
        if not content:
            continue
        # Truncate very long memory items to avoid blowout
        if len(content) > 600:
            content = content[:600].rsplit(" ", 1)[0] + " …"
        lines.append(f"- {content}")

    return "\n".join(lines) + "\n"


class ContextCompressor:
    """Assemble agent prompts with semantic compression and cache zones.

    This is a drop-in replacement for the raw ``ContextMemory`` call inside
    ``runner._prefixed``. It still delegates to ``ContextMemory`` for the
    3-tier degradation logic, but enriches *priority_context* with task-relevant
    memory and code headers before doing so.
    """

    def __init__(self, notes: Sequence[dict], *, recent_n: int = 10) -> None:
        self._memory = ContextMemory(notes, recent_n=recent_n)

    def build_prompt(
        self,
        *,
        role: str,
        priority_context: str,
        prompt: str,
        max_tokens: int,
        run_id: str | None = None,
        alias: str | None = None,
        summarize_fn: Callable[[str], str] | None = None,
        high_level: bool = False,
    ) -> str:
        """Build the final prompt string.

        Steps:
        1. Fetch relevant semantic context (summary only if high_level).
        2. Extract lightweight code context from the task prompt (signatures if high_level).
        3. Inject both into *priority_context*.
        4. Delegate to ``ContextMemory`` for tiered degradation.
        """
        relevant = _fetch_relevant_context(prompt, run_id, limit=(3 if high_level else 5))
        code_ctx = (
            _fetch_code_context(prompt, high_level=high_level)
            if config.compression_config().get("code_context_enabled", True)
            else ""
        )

        enriched_priority = priority_context
        if relevant:
            enriched_priority += relevant
        if code_ctx:
            enriched_priority += f"\n[CODE CONTEXT]\n{code_ctx}\n"

        return self._memory.build(
            role=role,
            priority_context=enriched_priority,
            prompt=prompt,
            max_tokens=max_tokens,
            token_counter=estimate_tokens,
            summarize_fn=summarize_fn,
        )
