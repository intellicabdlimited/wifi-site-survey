from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


PARAM_SCRIPT = Path(__file__).resolve().parent / "parameter_vs_range.py"



def patch_parameter_script(source_script: Path, patched_script: Path, param_display: str, input_dir: Path, out_dir: Path) -> Path:
    text = source_script.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"(?m)^PARAM_NAME\s*=.*$", f"PARAM_NAME = {param_display!r}", text, count=1)
    text = re.sub(r"(?m)^INPUT_DIR\s*=.*$", f"INPUT_DIR = {str(input_dir)!r}", text, count=1)
    text = re.sub(r"(?m)^OUT_BASE\s*=.*$", f"OUT_BASE = {str(out_dir)!r}", text, count=1)
    patched_script.parent.mkdir(parents=True, exist_ok=True)
    patched_script.write_text(text, encoding="utf-8")
    return patched_script



def run_multiple_parameters(
    parameters: Iterable[str],
    input_root: Path,
    output_root: Path,
    python_executable: str | None = None,
) -> List[dict]:
    results: List[dict] = []
    python_executable = python_executable or sys.executable
    generated_root = output_root / "_generated"
    generated_root.mkdir(parents=True, exist_ok=True)

    for param_display in parameters:
        metric_slug = re.sub(r"[^a-z0-9]+", "_", param_display.lower()).strip("_")
        metric_input = input_root / metric_slug
        metric_output = output_root / metric_slug
        patched_script = generated_root / f"parameter_vs_range__{metric_slug}.py"
        patch_parameter_script(PARAM_SCRIPT, patched_script, param_display, metric_input, metric_output)
        proc = subprocess.run(
            [python_executable, str(patched_script)],
            cwd=str(Path(__file__).resolve().parent),
            env={**os.environ, "RVR_INPUT_DIR": str(metric_input), "RVR_OUT_BASE": str(metric_output), "RVR_PARAM_NAME": param_display},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        results.append(
            {
                "parameter": param_display,
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "output_dir": str(metric_output),
            }
        )
    return results
