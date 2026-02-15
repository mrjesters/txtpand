"""CLI entry point: python -m txtpand

Subcommands:
    txtpand "cn y hel me"                  # one-shot expand
    txtpand pipe                           # interactive line-by-line expansion
    txtpand clip "cn y hel me"             # expand + copy to clipboard
    txtpand clip --watch                   # interactive expand + clipboard
    txtpand config                         # show / init config file
"""

from __future__ import annotations

import argparse
import sys

from txtpand.cli.factory import build_expander


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="txtpand",
        description="Expand shorthand text to full English. Type less, say more.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    sub = parser.add_subparsers(dest="command")

    # --- expand (default, also works without subcommand) ---
    expand_parser = sub.add_parser("expand", help="Expand text (default)")
    expand_parser.add_argument("text", nargs="?", help="Text to expand")
    expand_parser.add_argument("--spaceless", action="store_true")
    expand_parser.add_argument("--detailed", action="store_true")

    # --- pipe ---
    pipe_parser = sub.add_parser(
        "pipe",
        help="Interactive line-by-line expansion (stdin → stdout)",
    )
    pipe_parser.add_argument("--spaceless", action="store_true")

    # --- clip ---
    clip_parser = sub.add_parser(
        "clip",
        help="Expand text and copy to clipboard",
    )
    clip_parser.add_argument("text", nargs="?", help="Text to expand")
    clip_parser.add_argument("--spaceless", action="store_true")
    clip_parser.add_argument(
        "--watch", action="store_true",
        help="Interactive mode: type → enter → expanded + copied",
    )

    # --- config ---
    config_parser = sub.add_parser("config", help="Show or init config file")
    config_parser.add_argument(
        "--init", action="store_true",
        help="Create default config file at ~/.config/txtpand/config.toml",
    )

    # Intercept before argparse: if first arg isn't a known subcommand,
    # treat the entire input as text for "expand"
    raw_args = argv if argv is not None else sys.argv[1:]
    known_commands = {"expand", "pipe", "clip", "config"}

    if raw_args and raw_args[0] not in known_commands and not raw_args[0].startswith("-"):
        # Bare text: txtpand "cn y hel me"
        args = argparse.Namespace(
            command="expand",
            text=" ".join(raw_args),
            spaceless=False,
            detailed=False,
        )
    elif not raw_args and not sys.stdin.isatty():
        # Piped stdin with no args
        args = argparse.Namespace(
            command="expand",
            text=None,
            spaceless=False,
            detailed=False,
        )
    else:
        args = parser.parse_args(argv)
        if args.command is None:
            if not sys.stdin.isatty():
                args.command = "expand"
                args.text = None
                args.spaceless = False
                args.detailed = False
            else:
                parser.print_help()
                return

    if args.command == "expand":
        _cmd_expand(args)
    elif args.command == "pipe":
        _cmd_pipe(args)
    elif args.command == "clip":
        _cmd_clip(args)
    elif args.command == "config":
        _cmd_config(args)


def _cmd_expand(args: argparse.Namespace) -> None:
    """One-shot text expansion."""
    if args.text:
        text = args.text
    elif not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    else:
        print("Usage: txtpand \"your shorthand text\"", file=sys.stderr)
        print("       txtpand pipe     (interactive mode)", file=sys.stderr)
        print("       txtpand clip     (expand + clipboard)", file=sys.stderr)
        return

    if not text:
        return

    expander = build_expander()
    spaceless = getattr(args, "spaceless", False)
    detailed = getattr(args, "detailed", False)

    if detailed:
        report = expander.expand_detailed(text, spaceless=spaceless)
        print(f"Input:      {report.input}")
        print(f"Expanded:   {report.expanded}")
        print(f"Confidence: {report.confidence:.2f}")
        print(f"Elapsed:    {report.elapsed_ms:.1f}ms")
        if report.segments:
            print(f"Segments:   {report.segments}")
        print(f"LLM used:   {report.llm_used}")
        print()
        for tr in report.tokens:
            flag = ""
            if tr.ambiguous:
                flag = " [AMBIGUOUS]"
            if tr.llm_resolved:
                flag = " [LLM]"
            if tr.original != tr.expanded:
                print(f"  {tr.original!r} → {tr.expanded!r} ({tr.tier.value}, {tr.confidence:.2f}){flag}")
            else:
                print(f"  {tr.original!r} (passthrough)")
    else:
        result = expander.expand(text, spaceless=spaceless)
        print(result)


def _cmd_pipe(args: argparse.Namespace) -> None:
    """Interactive pipe mode."""
    from txtpand.cli.pipe import run_pipe

    run_pipe(spaceless=args.spaceless)


def _cmd_clip(args: argparse.Namespace) -> None:
    """Clipboard mode."""
    from txtpand.cli.clip import run_clip

    run_clip(text=args.text, spaceless=args.spaceless, watch=args.watch)


def _cmd_config(args: argparse.Namespace) -> None:
    """Config management."""
    from txtpand.cli.config_file import init_config_file, load_config

    if args.init:
        path = init_config_file()
        print(f"Config file created at: {path}")
        print(f"Edit it to add your API key and enable LLM polish.")
        return

    # Show current config
    config = load_config()
    print("txtpand configuration:")
    print(f"  LLM enabled:  {config.llm.enabled}")
    if config.llm.enabled:
        print(f"  Provider:     {config.llm.provider}")
        print(f"  Model:        {config.llm.model}")
        print(f"  API key:      {'***' + config.llm.api_key[-4:] if config.llm.api_key and len(config.llm.api_key) > 4 else '(not set)'}")
        print(f"  Timeout:      {config.llm.timeout}s")
    else:
        print("  (set OPENAI_API_KEY or run 'txtpand config --init' to enable)")
    print(f"  Spaceless:    {config.spaceless}")


if __name__ == "__main__":
    main()
