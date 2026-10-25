# Plan: Toolmaker sandbox hardening

## Status
Draft

## Goal
Close the verified AST-validator bypasses and move isolation enforcement from environment
variables to OS-level controls, so agent-authored tools cannot escape the sandbox.

## Context
Audit finding C-5 (docs/quality-audit.md). Verified gaps in `toolmaker/validator.py`:
`os` and `sys` are missing from `_BLOCKED_IMPORTS` (line 21) — `import os; os.system("curl …")`
passes validation and defeats the proxy-env-only network block in `sandbox.py`;
`open(f, mode_var)` bypasses the write-mode check (only `ast.Constant` modes inspected, ~line
104); `getattr(obj, "__globals__")` bypasses the attribute blocklist. This is a security
surface: toolmaker exists precisely to run unreviewed agent-generated code.

## Phases

### Phase 1: Close validator gaps (quick, high value)
- [ ] Add `os`, `sys`, `shutil`, `pathlib`(write APIs reviewed), `tempfile`(?) — decide the
  minimal allow-set rather than growing the blocklist; prefer an **allowlist** of importable
  modules (`json`, `re`, `math`, `datetime`, `collections`, `itertools`, `functools`, `typing`)
- [ ] Reject `open()` calls whose mode argument is not a constant; reject `getattr`/`setattr`/
  `delattr` with non-constant or dunder string attributes
- [ ] Adversarial test corpus: each known bypass becomes a red test
  (`tests/test_toolmaker_adversarial.py`)
- **Acceptance:** all bypass tests rejected at validation; existing legitimate tools still pass.

### Phase 2: OS-level enforcement
- [ ] macOS: wrap subprocess in `sandbox-exec` profile (no network, tmp-only writes);
  Linux: seccomp-bpf via `prctl` preexec or `unshare -n` network namespace
- [ ] Remove reliance on `http_proxy=127.0.0.1:0` (keep as defense-in-depth only)
- **Acceptance:** a validator-bypassing script (test-only) still cannot reach the network or
  write outside its temp dir.

### Phase 3: Decide the WASM path
- [ ] Evaluate `wasm_sandbox.py` (405 lines, experimental) as the primary isolation mechanism —
  WASI's capability model is categorically stronger than AST filtering
- [ ] ADR: `docs/decisions/adr-001-toolmaker-isolation.md` choosing subprocess+OS-sandbox vs WASM
- **Acceptance:** ADR accepted; losing path clearly marked experimental or removed.

## Risks
- Over-blocking breaks legitimate tools → run Phase 1 against the installed-tools corpus in
  `.orchestra/` state before merging.
- `sandbox-exec` is deprecated-but-functional on macOS → document fallback (ulimit + allowlist)
  if Apple removes it.
