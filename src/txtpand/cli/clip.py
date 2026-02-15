"""txtpand clip — expand text and copy to clipboard.

Type messy notes, press enter, get clean text pasted to clipboard.

Usage:
    # From argument
    txtpand clip "cn y hel me wo on smth"

    # From stdin
    echo "cn y hel me" | txtpand clip

    # Interactive mode — type, press enter, clipboard updated
    txtpand clip --watch
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from txtpand.cli.factory import build_expander


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to system clipboard. Returns True on success."""
    # WSL
    if shutil.which("clip.exe"):
        try:
            subprocess.run(
                ["clip.exe"],
                input=text.encode("utf-16-le"),
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            pass

    # macOS
    if shutil.which("pbcopy"):
        try:
            subprocess.run(
                ["pbcopy"],
                input=text.encode(),
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            pass

    # Linux with xclip
    if shutil.which("xclip"):
        try:
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode(),
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            pass

    # Linux with xsel
    if shutil.which("xsel"):
        try:
            subprocess.run(
                ["xsel", "--clipboard", "--input"],
                input=text.encode(),
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            pass

    # wl-copy (Wayland)
    if shutil.which("wl-copy"):
        try:
            subprocess.run(
                ["wl-copy"],
                input=text.encode(),
                check=True,
                timeout=5,
            )
            return True
        except (subprocess.SubprocessError, OSError):
            pass

    return False


def run_clip(
    text: str | None = None,
    spaceless: bool = False,
    watch: bool = False,
) -> None:
    """Run the clip command."""
    expander = build_expander()

    if watch:
        _run_watch_mode(expander, spaceless)
        return

    # Get input
    if text:
        input_text = text
    elif not sys.stdin.isatty():
        input_text = sys.stdin.read().strip()
    else:
        print("txtpand clip — expand and copy to clipboard", file=sys.stderr)
        print("Usage: txtpand clip \"your messy text\"", file=sys.stderr)
        print("       txtpand clip --watch  (interactive mode)", file=sys.stderr)
        return

    if not input_text:
        return

    expanded = expander.expand(input_text, spaceless=spaceless)
    print(expanded)

    if _copy_to_clipboard(expanded):
        print("(copied to clipboard)", file=sys.stderr)
    else:
        print("(clipboard not available — output printed above)", file=sys.stderr)


def _run_watch_mode(expander, spaceless: bool) -> None:
    """Interactive watch mode: type → enter → expanded + copied."""
    print(
        "txtpand clip watch — type text, press enter to expand + copy. Ctrl+C to quit.",
        file=sys.stderr,
    )
    if expander.config.llm_enabled:
        print("LLM polish: enabled", file=sys.stderr)
    print(file=sys.stderr)

    try:
        while True:
            sys.stderr.write("› ")
            sys.stderr.flush()

            try:
                line = input()
            except EOFError:
                break

            if not line.strip():
                continue

            expanded = expander.expand(line, spaceless=spaceless)
            print(f"  {expanded}")

            if _copy_to_clipboard(expanded):
                print("  (copied)", file=sys.stderr)

    except KeyboardInterrupt:
        print("\n", file=sys.stderr)
