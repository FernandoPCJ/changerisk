"""
CLI entrypoint for the multi-agent code audit.

Usage
-----
    python -m src.audit_agents.run_audit src/inference.py
    python -m src.audit_agents.run_audit src/run_szz.py --out reports/run_szz_audit.md

Requires:
- the extra dependencies in requirements-agents.txt (`pip install -r requirements-agents.txt`)
- a running Ollama instance with the model set in AUDIT_LLM_MODEL
  pulled locally (default: ollama/deepseek-r1:14b), OR
- OPENAI_API_KEY / ANTHROPIC_API_KEY set and AUDIT_LLM_MODEL pointed
  at that provider instead (e.g. "openai/gpt-4o").
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.audit_agents.crew import run_audit


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Path to the Python file to audit")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the markdown report (default: reports/<file>_audit.md)",
    )
    args = parser.parse_args()

    file_path = Path(args.file)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path("reports") / f"{file_path.stem}_audit.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Auditing {file_path} ...")
    report = run_audit(str(file_path))

    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
