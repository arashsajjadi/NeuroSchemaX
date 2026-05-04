"""Command-line interface for NeuroSchemaX."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .api.export import (
    save_debug_json_from,
    save_nnsvg_spec,
    save_paper_json_from,
)
from .api.parse import parse_model
from .api.render import (
    build_nnsvg_spec,
    recommend_view,
    save_network_html,
    save_network_svg,
)
from .api.summarize import summarize_model
from .exceptions import NeuroSchemaXError
from .version import __version__
from .visualization.compat import environment_summary


def _add_common_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="Path to the model file (.onnx, .json, .yaml)")


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", required=True, help="Output file path (.html or .svg)")
    parser.add_argument("--format", default=None, choices=["html", "svg"],
                        help="Output format (inferred from extension if omitted)")
    parser.add_argument("--theme", default=None,
                        choices=["paper", "thesis", "debug", "readme"],
                        help="Visual theme preset (default: paper)")
    parser.add_argument("--style", default=None,
                        choices=["fcnn", "lenet", "alexnet"],
                        help="Force a specific NN-SVG diagram family (default: auto-detect)")
    parser.add_argument("--width", type=int, default=None,
                        metavar="PX", help="Canvas width in pixels (default: 1200–auto)")
    parser.add_argument("--height", type=int, default=None,
                        metavar="PX", help="Canvas height in pixels (default: 700)")
    parser.add_argument("--title", default=None,
                        help="Diagram title shown above the diagram")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide layer name labels")
    parser.add_argument("--no-shapes", action="store_true",
                        help="Hide shape/dimension annotations on labels")
    parser.add_argument("--compact", action="store_true",
                        help="Use compact layout (useful for large models)")


def _add_draw_render_args(parser: argparse.ArgumentParser) -> None:
    """Render args for the draw command (output is optional)."""
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (.html or .svg); default: <model>.html")
    parser.add_argument("--theme", default=None,
                        choices=["paper", "thesis", "debug", "readme"],
                        help="Visual theme preset (default: paper)")
    parser.add_argument("--style", default=None,
                        choices=["fcnn", "lenet", "alexnet"],
                        help="Force a specific NN-SVG diagram family (default: auto-detect)")
    parser.add_argument("--width", type=int, default=None,
                        metavar="PX", help="Canvas width in pixels (default: auto)")
    parser.add_argument("--height", type=int, default=None,
                        metavar="PX", help="Canvas height in pixels (default: 700)")
    parser.add_argument("--title", default=None,
                        help="Diagram title shown above the diagram")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide layer name labels")
    parser.add_argument("--no-shapes", action="store_true",
                        help="Hide shape/dimension annotations on labels")
    parser.add_argument("--compact", action="store_true",
                        help="Use compact layout (useful for large models)")


def _kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    if args.theme:
        kwargs["theme"] = args.theme
    if args.style:
        kwargs["style"] = args.style
    if args.width:
        kwargs["width"] = args.width
    if args.height:
        kwargs["height"] = args.height
    if args.title:
        kwargs["title"] = args.title
    if args.no_labels:
        kwargs["show_labels"] = False
    if args.no_shapes:
        kwargs["show_shapes"] = False
    if args.compact:
        kwargs["compact"] = True
    return kwargs


# ── Command implementations ──────────────────────────────────────────────────

def cmd_inspect(args: argparse.Namespace) -> int:
    arch = parse_model(args.model)
    print(f"Model:            {arch.model_name}")
    print(f"Framework:        {arch.framework}")
    print(f"Layer count:      {arch.layer_count}")
    print(f"Block count:      {len(arch.blocks)}")
    if arch.recommended_family:
        approx = " [approximate]" if arch.warnings else ""
        print(
            f"Recommended view: {arch.recommended_family.value}"
            f" (confidence: {arch.family_confidence.value}){approx}"
        )
    if arch.warnings:
        print("\nWarnings:")
        for w in arch.warnings:
            # Wrap long warnings at 80 chars
            print(f"  ! {w[:120]}" + ("..." if len(w) > 120 else ""))
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    out = summarize_model(args.model, format=args.format)
    print(out)
    return 0


def cmd_recommend_view(args: argparse.Namespace) -> int:
    info = recommend_view(args.model)
    # JSON to stdout (machine-readable, pipe-friendly)
    print(json.dumps(info, indent=2))
    # Human-readable summary to stderr (interactive display)
    approx_flag = " [APPROXIMATE]" if info.get("is_approximate") else ""
    print(f"Family: {info['family']}  Confidence: {info['confidence']}{approx_flag}",
          file=sys.stderr)
    print(f"Reason: {info['reason']}", file=sys.stderr)
    if info.get("warnings"):
        print("Warnings:", file=sys.stderr)
        for w in info["warnings"]:
            print(f"  - {w[:100]}" + ("..." if len(w) > 100 else ""), file=sys.stderr)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    output = Path(args.output)
    fmt = args.format
    if fmt is None:
        suffix = output.suffix.lower()
        if suffix == ".html":
            fmt = "html"
        elif suffix == ".svg":
            fmt = "svg"
        else:
            print(f"error: cannot infer format from extension {suffix!r}",
                  file=sys.stderr)
            return 2

    kwargs = _kwargs_from_args(args)
    if fmt == "html":
        save_network_html(output, args.model, **kwargs)
    else:
        save_network_svg(output, args.model, **kwargs)
    print(f"Wrote {output}")
    return 0


def cmd_draw(args: argparse.Namespace) -> int:
    """User-friendly alias for render. Output defaults to <model_stem>.html."""
    model_path = Path(args.model)

    if args.output is None:
        output = Path.cwd() / (model_path.stem + ".html")
        # Defer the informational message to stderr so it doesn't clutter stdout
        # and only appears after we know the render will proceed.
        default_output = True
    else:
        output = Path(args.output)
        default_output = False

    suffix = output.suffix.lower()
    if suffix == ".html":
        fmt = "html"
    elif suffix == ".svg":
        fmt = "svg"
    else:
        fmt = "html"
        output = output.with_suffix(".html")
        print(f"note: unrecognised extension; saving as HTML to: {output}",
              file=sys.stderr)

    kwargs = _kwargs_from_args(args)
    if fmt == "html":
        save_network_html(output, args.model, **kwargs)
    else:
        save_network_svg(output, args.model, **kwargs)

    if default_output:
        print(f"Wrote {output}  (use -o to choose a different path)")
    else:
        print(f"Wrote {output}")
    return 0


def cmd_export_paper(args: argparse.Namespace) -> int:
    save_paper_json_from(args.output, args.model)
    print(f"Wrote {args.output}")
    return 0


def cmd_export_debug(args: argparse.Namespace) -> int:
    save_debug_json_from(args.output, args.model)
    print(f"Wrote {args.output}")
    return 0


def cmd_export_nnsvg(args: argparse.Namespace) -> int:
    kwargs = _kwargs_from_args(args)
    spec = build_nnsvg_spec(args.model, **kwargs)
    save_nnsvg_spec(args.output, spec)
    print(f"Wrote {args.output}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    info = environment_summary()

    # Print JSON so the output is machine-parseable (backward compatible)
    print(json.dumps(info, indent=2))

    # Also print a human-readable summary to stderr for interactive use
    import sys as _sys
    status = info.get("status", "unknown")
    status_label = {"ok": "OK", "partial": "PARTIAL", "error": "ERROR"}.get(
        status, status.upper()
    )
    print(f"\n--- doctor summary [status: {status_label}] ---", file=_sys.stderr)
    print(f"  Version : {info.get('version', 'unknown')}", file=_sys.stderr)
    print(f"  Python  : {info.get('python', 'unknown')}", file=_sys.stderr)

    assets = info.get("assets", {})
    print("\n  Assets:", file=_sys.stderr)
    for name, present in assets.items():
        mark = "PASS" if present else "FAIL"
        print(f"    [{mark}] {name}", file=_sys.stderr)

    deps = info.get("dependencies", {})
    print("\n  Dependencies:", file=_sys.stderr)
    for dep, present in deps.items():
        mark = "PASS" if present else "MISS"
        print(f"    [{mark}] {dep}", file=_sys.stderr)

    messages = info.get("messages", [])
    if messages:
        print("\n  Action items:", file=_sys.stderr)
        for msg in messages:
            print(f"    - {msg}", file=_sys.stderr)
    else:
        print("\n  All checks passed.", file=_sys.stderr)

    return 0


# ── Parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuroschemax",
        description="Neural network architecture visualization & export toolkit "
                    "(powered by NN-SVG).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # inspect
    p = sub.add_parser("inspect", help="Print a brief structural overview")
    _add_common_input_args(p)
    p.set_defaults(func=cmd_inspect)

    # summarize
    p = sub.add_parser("summarize", help="Print a summary of the model")
    _add_common_input_args(p)
    p.add_argument("--format", default="text", choices=["text", "markdown"])
    p.set_defaults(func=cmd_summarize)

    # recommend-view
    p = sub.add_parser("recommend-view", help="Recommend a rendering style")
    _add_common_input_args(p)
    p.set_defaults(func=cmd_recommend_view)

    # render
    p = sub.add_parser("render", help="Render a model to HTML or SVG")
    _add_common_input_args(p)
    _add_render_args(p)
    p.set_defaults(func=cmd_render)

    # draw (user-friendly alias for render)
    p = sub.add_parser(
        "draw",
        help="Draw a model diagram (friendly alias for render; output defaults to <model>.html)",
    )
    _add_common_input_args(p)
    _add_draw_render_args(p)
    p.set_defaults(func=cmd_draw)

    # export-paper-json
    p = sub.add_parser("export-paper-json", help="Export paper-oriented JSON")
    _add_common_input_args(p)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_export_paper)

    # export-debug-json
    p = sub.add_parser("export-debug-json", help="Export verbose debug JSON")
    _add_common_input_args(p)
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_export_debug)

    # export-nnsvg
    p = sub.add_parser("export-nnsvg", help="Export the NN-SVG JSON spec")
    _add_common_input_args(p)
    p.add_argument("-o", "--output", required=True)
    _add_render_args_for_export(p)
    p.set_defaults(func=cmd_export_nnsvg)

    # doctor
    p = sub.add_parser("doctor", help="Show environment / asset diagnostics")
    p.set_defaults(func=cmd_doctor)

    return parser


def _add_render_args_for_export(parser: argparse.ArgumentParser) -> None:
    """A subset of render args, usable without -o duplication."""
    parser.add_argument("--theme", default=None,
                        choices=["paper", "thesis", "debug", "readme"])
    parser.add_argument("--style", default=None,
                        choices=["fcnn", "lenet", "alexnet"])
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--no-shapes", action="store_true")
    parser.add_argument("--compact", action="store_true")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except NeuroSchemaXError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
