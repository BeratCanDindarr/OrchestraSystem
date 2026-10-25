"""The video_frames tasks for the CLI agents (Claude, Kimi, Gemini through agy).

bench.py sends images to Ollama models only, so here the sheet is copied into the CLIs'
working directory and the prompt names the file for the agent to open.

    python tools/model_bench/bench.py tools/model_bench/tasks/video_frames_cli.py \
        --models claude-opus,claude-sonnet,kimi-k3,kimi-coding,gemini-4-argon,gemini-3.8-flash
"""
from __future__ import annotations

import importlib.util
import os
import shutil

_spec = importlib.util.spec_from_file_location(
    "video_frames", os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_frames.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

CLI_CWD = "/tmp/model_bench_cwd"  # same as bench.CLI_CWD
os.makedirs(CLI_CWD, exist_ok=True)
for _name, _path in _base.IMAGES.items():
    shutil.copy(_path, os.path.join(CLI_CWD, f"sheet_{_name}.png"))

PROMPTS = {
    name: f"Open and look at the image file ./sheet_{name}.png in the current directory. " + prompt
    for name, prompt in _base.PROMPTS.items()
}
SCORERS = _base.SCORERS
