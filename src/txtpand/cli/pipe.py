"""txtpand pipe — interactive line-by-line expansion.

Reads stdin line by line, expands each, writes to stdout.
Use it as a pre-processor before sending text to AI tools.

Usage:
    # Interactive — type messy, see clean
    txtpand pipe

    # Pipe to another tool
    txtpand pipe | some-ai-cli

    # One-shot from file
    cat notes.txt | txtpand pipe > clean_notes.txt
"""

from __future__ import annotations

import sys

from txtpand.cli.factory import build_expander


def run_pipe(spaceless: bool = False) -> None:
    """Run the pipe command — read lines from stdin, expand, write to stdout."""
    expander = build_expander()
    interactive = sys.stdin.isatty()

    if interactive:
        print("txtpand pipe — type text, get expanded output. Ctrl+C to quit.", file=sys.stderr)
        if expander.config.llm_enabled:
            print("LLM polish: enabled", file=sys.stderr)
        print(file=sys.stderr)

    try:
        while True:
            if interactive:
                sys.stderr.write("› ")
                sys.stderr.flush()

            try:
                line = input()
            except EOFError:
                break

            if not line.strip():
                if interactive:
                    print()
                continue

            expanded = expander.expand(line, spaceless=spaceless)
            print(expanded)
            sys.stdout.flush()
    except KeyboardInterrupt:
        if interactive:
            print("\n", file=sys.stderr)
