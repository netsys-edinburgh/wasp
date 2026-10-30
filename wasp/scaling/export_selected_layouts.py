#!/usr/bin/env python3
"""A3: derive and export the selected DP/PP/TP (or pipeline-stage) layout that
each baseline's planner chose for each principal model/device configuration.

The layout is derived from the dispatch manifests (the planners' outputs):
pipeline degree = number of distinct stage_ids; the within-stage device count
splits into TP (if a tp_allreduce collective is present, e.g. Alpa),
DP (if a dp_allreduce collective is present, e.g. DTFM/Asteroid), or pure
pipeline (Confidant). WASP dispatches sub-GEMM shards under selective TP across
the whole fleet. Emits results/analytical_scaling/selected_layouts.json and a
LaTeX table body.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BASELINES = ["cleave", "dtfm", "asteroid", "confident", "alpa"]
LABELS = {
    "cleave": "WASP",
    "dtfm": "DTFM",
    "asteroid": "Asteroid",
    "confident": "Confidant",
    "alpa": "Alpa",
}


def derive_layout(manifest: dict[str, Any]) -> dict[str, Any]:
    ents = manifest.get("entries", [])
    meta = manifest.get("metadata", {})
    nd = int(meta.get("num_devices", 0)) or len(
        {
            int(e.get("device_id", -1))
            for e in ents
            if int(e.get("device_id", -1)) >= 0
        }
    )
    ptypes = {str(e.get("parallelism_type", e.get("type", ""))) for e in ents}

    if "cleave_tp" in ptypes:
        return {
            "num_devices": nd,
            "granularity": "sub-GEMM shard",
            "dp": 1,
            "pp": 1,
            "tp": nd,
            "summary": f"selective sub-GEMM TP across {nd}",
        }

    stages = sorted({int(e.get("stage_id", -1)) for e in ents})
    pp = max(1, len(stages))
    dev_per_stage: dict[int, set[int]] = {}
    for e in ents:
        dev_per_stage.setdefault(int(e.get("stage_id", -1)), set()).add(
            int(e.get("device_id", -1))
        )
    per_stage = max((len(v) for v in dev_per_stage.values()), default=nd)

    has_dp = any("dp_allreduce" in p for p in ptypes)
    has_tp = any("tp_allreduce" in p for p in ptypes)

    if has_tp:
        tp, dp = per_stage, 1
    elif has_dp:
        dp, tp = per_stage, 1
    else:
        dp, tp = (per_stage, 1) if per_stage > 1 else (1, 1)

    parts = []
    if dp > 1:
        parts.append(f"DP{dp}")
    if pp > 1:
        parts.append(f"PP{pp}")
    if tp > 1:
        parts.append(f"TP{tp}")
    summary = " x ".join(parts) if parts else "PP1"
    return {
        "num_devices": nd,
        "granularity": "pipeline/layer",
        "dp": dp,
        "pp": pp,
        "tp": tp,
        "summary": summary,
    }


def _manifest_path(
    root: Path, cfg: dict[str, str], baseline: str
) -> Path | None:
    for pat in cfg["patterns"]:
        p = root / pat.format(b=baseline)
        if p.exists():
            return p
    return None


def build_configs(root: Path) -> dict[str, dict[str, str]]:
    configs: dict[str, dict[str, str]] = {}
    for n, pt in [
        (64, "000_64"),
        (128, "001_128"),
        (256, "002_256"),
        (512, "003_512"),
        (1024, "004_1024"),
    ]:
        configs[f"opt-13b/{n}"] = {
            "patterns": [
                f"results/vtime_scaling/num_devices/points/{pt}/planning/manifests/{{b}}_manifest.json"
            ]
        }
    configs["llama2-13b"] = {
        "patterns": [
            "results/vtime_models/llama2-13b/{b}_results/manifests/{b}_manifest.json"
        ]
    }
    configs["llama2-70b"] = {
        "patterns": [
            "results/vtime_models/llama2-70b/{b}_results/manifests/{b}_manifest.json"
        ]
    }
    return configs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument(
        "--output",
        type=str,
        default="results/analytical_scaling/selected_layouts.json",
    )
    args = ap.parse_args()
    root = Path(args.root)

    configs = build_configs(root)
    out: dict[str, Any] = {
        "description": "Selected planner layouts per (config, baseline), derived from dispatch manifests.",
        "configs": {},
    }
    for cfg_name, cfg in configs.items():
        cfg_out: dict[str, Any] = {}
        for b in BASELINES:
            mp = _manifest_path(root, cfg, b)
            if mp is None:
                continue
            cfg_out[b] = derive_layout(json.loads(mp.read_text()))
        if cfg_out:
            out["configs"][cfg_name] = cfg_out

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"Wrote {outp}\n")
    for cfg_name, cfg_out in out["configs"].items():
        row = "  ".join(
            f"{LABELS[b]}={v['summary']}({v['num_devices']}d)"
            for b, v in cfg_out.items()
        )
        print(f"{cfg_name:16s}: {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
