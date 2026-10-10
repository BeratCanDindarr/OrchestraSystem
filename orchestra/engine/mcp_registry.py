from __future__ import annotations
import inspect
import logging
import json
import sys
import traceback
import typing
from typing import Any, Callable, Optional, Union, get_type_hints

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type-annotation helpers
# ---------------------------------------------------------------------------

_PY_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
}


def _unwrap_optional(annotation: Any) -> Any:
    """Unwrap ``Optional[X]`` (i.e. ``Union[X, None]``) to ``X``.

    Returns the annotation unchanged if it is not an Optional wrapper.
    This handles both ``typing.Optional`` and ``X | None`` syntax under
    Python 3.10+.
    """
    origin = getattr(annotation, "__origin__", None)
    # typing.Optional[X] is represented as Union[X, NoneType]
    if origin is Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string.

    Unwraps ``Optional[X]`` before mapping so ``Optional[bool]`` → "boolean".
    """
    unwrapped = _unwrap_optional(annotation)
    return _PY_TO_JSON_TYPE.get(unwrapped, "string")


def _derive_schema(fn: Callable) -> dict:
    """Auto-derive a minimal JSON Schema ``inputSchema`` from *fn*'s signature.

    Rules:
    - Each parameter becomes an object property.
    - Type derived from annotation; missing annotation → "string".
    - Parameters WITHOUT defaults go into the ``required`` list.
    - *args / **kwargs are skipped.

    Uses ``typing.get_type_hints()`` to resolve stringified annotations that
    result from ``from __future__ import annotations`` (PEP 563), so that
    ``bool`` and ``int`` parameters in such modules map correctly to
    "boolean" and "integer" rather than falling back to "string".
    """
    sig = inspect.signature(fn)
    # Resolve stringified annotations produced by PEP 563.  Fall back to the
    # raw inspect annotations if the module lacks the necessary globals.
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        # Prefer resolved hint; fall back to raw annotation from inspect
        resolved = hints.get(name, param.annotation)
        if resolved is inspect.Parameter.empty:
            json_t = "string"
        else:
            json_t = _json_type(resolved)
        properties[name] = {"type": json_t}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _scrub_dict(payload: dict) -> dict:
    from orchestra.engine.pii_scrubber import PiiScrubber
    return PiiScrubber.scrub_dict(payload)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _tool_success(payload: dict) -> dict:
    scrubbed = _scrub_dict(payload)
    return {
        "content": [{"type": "text", "text": _json_dumps(scrubbed)}],
        "structuredContent": scrubbed,
        "isError": False,
    }


def _tool_error(message: str) -> dict:
    return {
        "content": [{"type": "text", "text": str(message)}],
        "isError": True,
    }


def _structured_run_result(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    run = payload.get("run")
    if not isinstance(run, dict):
        run = {}

    return {
        "run_id": str(run.get("run_id", "")),
        "status": str(run.get("status", "unknown")),
        "mode": str(run.get("mode", "unknown")),
        "agent_count": len(run.get("agents", []) or []),
        "approval_state": str(run.get("approval_state", "not_required")),
        "waiting_approval": bool(run.get("status") == "waiting_approval"),
    }


TASK_ECHO_CHARS = 200
_JOB_KEYS = ("job_id", "run_id", "status", "mode", "alias", "follow_up_to", "error_code", "error_message")
_RUN_KEYS = ("run_id", "mode", "status", "summary", "total_cost_usd", "avg_confidence", "latest_review_status",
             "latest_review_reason", "approval_state", "failure", "parent_run_id", "follows_run_id", "created_at", "updated_at")
_AGENT_KEYS = ("alias", "provider", "model", "status", "elapsed", "error", "estimated_cost_usd",
               "confidence", "validation_status", "soft_failed")


def _short_task(task: Any) -> str:
    text = str(task or "")
    return text if len(text) <= TASK_ECHO_CHARS else text[:TASK_ECHO_CHARS] + " …"


def _compact_run_payload(payload: dict) -> dict:
    """What a caller needs from a run tool: the answer, who gave it, the state and the cost.

    The service payload carries the whole manifest, the job row and every event, and the
    task text several times over; the caller wrote the task and can ask orchestra_get_run
    for the rest.
    """
    compact: dict = {key: value for key, value in payload.items() if key not in ("job", "run", "events", "result")}
    job = payload.get("job")
    if isinstance(job, dict):
        compact["job"] = {key: job[key] for key in _JOB_KEYS if job.get(key) is not None}
    run = payload.get("run")
    if isinstance(run, dict):
        kept = {key: run[key] for key in _RUN_KEYS if run.get(key) not in (None, "", [])}
        kept["task"] = _short_task(run.get("task"))
        kept["agents"] = [{key: agent[key] for key in _AGENT_KEYS if agent.get(key) is not None}
                          for agent in run.get("agents") or [] if isinstance(agent, dict)]
        compact["run"] = kept
    events = payload.get("events")
    if isinstance(events, list):
        reasons = [event.get("reason") for event in events
                   if isinstance(event, dict) and event.get("event") == "approval_requested" and event.get("reason")]
        if reasons:
            compact["approval_reason"] = reasons[-1]
        # An alias that could not answer was replaced by its fallback: the caller must know
        # that two "independent" agents may have been the same model.
        fallbacks = {event.get("alias"): event.get("model") for event in events
                     if isinstance(event, dict) and event.get("event") == "agent_fallback"}
        if fallbacks:
            compact["fallbacks"] = [{"alias": alias, "answered_by": model} for alias, model in fallbacks.items()]
    return compact


def _tool_success_run(payload: dict, compact: bool = True) -> dict:
    if not isinstance(payload, dict):
        return _tool_error("Tool returned non-dict payload")
    structured = _compact_run_payload(payload) if compact else dict(payload)
    structured["result"] = _structured_run_result(payload)
    return _tool_success(structured)


_RUN_TOOLS = {
    "orchestra_submit_run",
    "orchestra_run",
    "orchestra_continue",
    "orchestra_continue_run",
    "orchestra_wait_for_run",
    "orchestra_execute_job",
    "orchestra_get_run",
    "orchestra_review_changes",
}


# The tool for reading everything about a run: its payload is returned whole.
_FULL_DETAIL_TOOLS = {"orchestra_get_run"}


def _wrap_result(name: str, result: Any) -> dict:
    if isinstance(result, dict):
        if name in _RUN_TOOLS:
            return _tool_success_run(result, compact=name not in _FULL_DETAIL_TOOLS)
        return _tool_success(result)
    return _tool_success({"result": result})


# ---------------------------------------------------------------------------
# Internal marker used by _auto_wrap to tag its closures
# ---------------------------------------------------------------------------

_AUTOWRAP_ATTR = "_orchestra_autowrapped"


def _auto_wrap(fn: Callable) -> Callable:
    """Wrap *fn* so it accepts a single ``arguments: dict`` and extracts kwargs.

    If *fn* is already an ``_auto_wrap`` closure (carries ``_AUTOWRAP_ATTR``),
    it is returned as-is to avoid double-wrapping.

    Logs a WARNING when the caller passes keys not in the function signature.
    """
    # Idempotent: don't double-wrap
    if getattr(fn, _AUTOWRAP_ATTR, False):
        return fn

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    def wrapper(arguments: dict) -> Any:
        if not isinstance(arguments, dict):
            arguments = {}
        unknown = [k for k in arguments if k not in params]
        if unknown:
            logger.warning(
                "tool %s ignoring unknown args: %s",
                fn.__name__,
                unknown,
            )
        kwargs = {p: arguments[p] for p in params if p in arguments}
        return fn(**kwargs)

    setattr(wrapper, _AUTOWRAP_ATTR, True)
    # Preserve original fn reference for schema derivation
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class McpRegistry:
    """Registry of MCP tools with schema metadata."""

    def __init__(self) -> None:
        # name -> (callable_accepting_dict, fastlane, description, input_schema)
        self._tools: dict[str, tuple[Callable, bool, str, dict]] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        *,
        fastlane: bool = True,
        description: str = "",
        input_schema: dict | None = None,
    ) -> None:
        """Register *fn* under *name*.

        *fn* may be either:

        a) A raw service function (keyword-arg style) — ``register`` wraps it
           with ``_auto_wrap`` and derives the schema from its original signature.
        b) A pre-wrapped closure produced by ``_auto_wrap`` (carries the
           ``_AUTOWRAP_ATTR`` marker) — used as-is; schema derived from the
           stored ``__wrapped__`` attribute when available.
        c) An ad-hoc callback that accepts a single ``arguments: dict`` positional
           parameter (legacy tests / inline lambdas) — detected by signature
           inspection and used as-is (no double-wrapping).

        Design rationale: accepting the raw service function as the preferred
        path means ``register`` can derive accurate schemas without exposing the
        wrapping detail to call sites.  The backward-compat path handles the
        pre-existing call-sites that passed ``_auto_wrap(fn)`` explicitly.
        """
        if name in self._tools:
            logger.warning("tool %s re-registered (overwriting previous entry)", name)

        # Determine whether fn is already callable(arguments: dict) style
        already_wrapped = getattr(fn, _AUTOWRAP_ATTR, False)

        # Detect legacy ad-hoc callbacks: single positional param whose name is
        # "arguments" OR whose annotation resolves to ``dict``.
        #
        # The old heuristic (any single-positional-param function) was too broad:
        # it matched real service functions like ``get_run(run_id: str)`` and
        # passed the whole arguments dict as the value of ``run_id``.
        #
        # New rule (opt-in legacy detection):
        #   A function is treated as a legacy ``(arguments: dict)`` callback ONLY
        #   when its sole positional parameter satisfies at least one of:
        #     1. The parameter is named "arguments".
        #     2. The parameter is annotated as ``dict`` (resolved via get_type_hints).
        #
        # Everything else is a normal keyword-arg function and gets _auto_wrap'd.
        if not already_wrapped:
            try:
                sig = inspect.signature(fn)
                params = [
                    p for p in sig.parameters.values()
                    if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                ]
                if len(params) == 1 and params[0].kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    sole = params[0]
                    # Resolve the annotation (handles PEP 563 stringified forms)
                    try:
                        hints = get_type_hints(fn)
                        resolved_type = hints.get(sole.name, inspect.Parameter.empty)
                    except Exception:
                        resolved_type = sole.annotation
                    annotation_is_dict = resolved_type is dict
                    annotation_absent = resolved_type is inspect.Parameter.empty
                    # Legacy callbacks use a "bag of arguments" parameter.
                    # Treat as legacy when EITHER:
                    #   a) The sole param is annotated as ``dict``, OR
                    #   b) The param is named "arguments" or "args" and is either
                    #      unannotated or annotated as ``dict`` (guards against
                    #      mis-detecting ``def f(args: str)``)
                    name_is_legacy = sole.name in ("arguments", "args")
                    is_single_dict_callback = annotation_is_dict or (
                        name_is_legacy and (annotation_absent or annotation_is_dict)
                    )
                else:
                    is_single_dict_callback = False
            except (TypeError, ValueError):
                is_single_dict_callback = False
        else:
            is_single_dict_callback = False

        # Determine the raw function for schema derivation
        if already_wrapped:
            raw_fn = getattr(fn, "__wrapped__", None)
        elif is_single_dict_callback:
            raw_fn = None  # cannot derive meaningful schema from (args: dict) sig
        else:
            raw_fn = fn

        # Derive or use provided schema
        if input_schema is not None:
            effective_schema = input_schema
        elif raw_fn is not None:
            try:
                effective_schema = _derive_schema(raw_fn)
            except (TypeError, ValueError):
                effective_schema = {"type": "object", "properties": {}}
        else:
            effective_schema = {"type": "object", "properties": {}}

        # Build the final callable
        if already_wrapped or is_single_dict_callback:
            callable_fn = fn
        else:
            callable_fn = _auto_wrap(fn)

        self._tools[name] = (callable_fn, fastlane, description, effective_schema)

    def invoke(self, name: str, arguments: dict) -> dict | None:
        entry = self._tools.get(name)
        if entry is None:
            return None
        fn, _fl, _desc, _schema = entry
        try:
            if not isinstance(arguments, dict):
                arguments = {}
            result = fn(arguments)
            return _wrap_result(name, result)
        except Exception as exc:
            err_msg = f"Error in {name}: {exc}"
            sys.stderr.write(f"FAIL {name}: {exc}\n{traceback.format_exc()}\n")
            return _tool_error(err_msg)

    def list_tools(self) -> list[dict]:
        """Return MCP tool descriptors sorted by name."""
        descriptors = []
        for name in sorted(self._tools):
            _fn, _fl, description, schema = self._tools[name]
            descriptors.append({
                "name": name,
                "description": description,
                "inputSchema": schema,
            })
        return descriptors

    def is_fastlane(self, name: str) -> bool:
        entry = self._tools.get(name)
        if entry is None:
            return False
        return entry[1]


REGISTRY = McpRegistry()
