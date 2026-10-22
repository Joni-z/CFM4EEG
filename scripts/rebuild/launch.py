"""Run a supported original-author entrypoint with a matching smoke gate.

Conformer, LaBraM, native23 EEGPT and TFM use explicit CLIs in docs/UPSTREAM.md.
There is no generic-baseline fallback.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("seed", type=int, choices=(0,))
    parser.add_argument("mode", choices=("smoke", "train"))
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get("disabled_reason"):
        parser.error(cfg["disabled_reason"])
    runner = cfg.get("runner")
    standard = {
        "scripts/rebuild/native_biot_entry.py",
        "scripts/rebuild/native_cbramod_entry.py",
        "scripts/rebuild/native_cbramod_clinical_entry.py",
        "scripts/rebuild/run_reve_native.py",
        "scripts/rebuild/run_eegpt_sleep_native.py",
    }
    distributed = "scripts/rebuild/run_eegpt_chb_official.py"
    if runner not in standard | {distributed}:
        parser.error("Use the model's explicit native CLI in docs/UPSTREAM.md; "
                     "generic baseline training is disabled.")
    output = (Path(os.environ["CFM_REBUILD_ROOT"])
              / ("smoke" if args.mode == "smoke" else "runs") / cfg["name"] / "seed0")
    gate = [sys.executable, "scripts/rebuild/preflight.py", "--config", str(args.config)]
    subprocess.run(gate + ["--mode", args.mode], check=True)
    command = [sys.executable, "-u"]
    if runner == distributed:
        command += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
    command += [runner, str(args.config), "--seed", "0", "--out", str(output)]
    if args.mode == "smoke":
        command += ["--smoke-steps", "4"]
    subprocess.run(command, check=True)
    if args.mode == "smoke":
        receipt = json.loads((output / "smoke.json").read_text())
        if (receipt.get("status") != "passed" or receipt.get("config") != cfg
                or receipt.get("checkpoint_reload") != "passed"
                or receipt.get("test_evaluated") is not False):
            raise RuntimeError("Native smoke identity, reload or test-access check failed")
        # Compare original recorded hashes; never replace them with new ones.
        subprocess.run(gate + ["--mode", "train"], check=True)


if __name__ == "__main__":
    main()
