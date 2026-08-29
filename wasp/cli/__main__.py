"""Wasp command-line interface."""

from __future__ import annotations

import argparse

from wasp.placement.backend import select_backend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wasp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="Print environment and solver backend")

    run = sub.add_parser("run", help="Run the emulated OPT-125M example")
    run.add_argument("--model-name", default="facebook/opt-125m")
    run.add_argument("--coords", type=int, default=2)
    run.add_argument("--devices-per-coord", type=int, default=1)
    run.add_argument("--steps", type=int, default=3)
    run.add_argument("--tiny", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "info":
        print(f"solver backend: {select_backend()}")
        return 0
    if args.command == "run":
        from wasp.coordinator.run_multi_coordinator import main as run_main

        return int(
            run_main(
                [
                    "--coords", str(args.coords),
                    "--devices_per_coord", str(args.devices_per_coord),
                    "--steps", str(args.steps),
                    "--model-name", args.model_name,
                    *(["--tiny"] if args.tiny else []),
                ]
            )
            or 0
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
