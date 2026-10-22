"""REVE electrode-bank lookup, extracted unchanged from the former adapter.

Bipolar coordinates use the released midpoint rule. The existing19/16-channel
fallbacks are retained for reproducibility; they do not establish unknown raw
channel order. See docs/DATA.md for the dataset contracts.
"""
import json
import os
import torch
import yaml
from paclock_bench.paths import REPO, vendored

POSITIONS_JSON = os.path.join(os.path.dirname(vendored("reve")), "reve_positions", "positions.json")

MONO19 = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
          "F7", "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
TUH16 = ["FP1-F7", "F7-T3", "T3-T5", "T5-O1", "FP2-F8", "F8-T4", "T4-T6",
         "T6-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4",
         "C4-P4", "P4-O2"]

def positions_for(dataset: str, n_channels: int, channels_override=None) -> torch.Tensor:
    """Coordinates under the existing corpus channel conventions.

    Explicit channel lists take priority. The16/19-channel defaults preserve the
    prior input contract; channel count does not prove raw electrode identity.
    """
    bank_raw = json.load(open(POSITIONS_JSON))
    bank = {k.upper(): v for k, v in bank_raw.items()}
    path = os.path.join(REPO, "configs", "datasets", f"{dataset}.yaml")
    channels = channels_override
    if channels is None and os.path.exists(path):
        channels = yaml.safe_load(open(path)).get("channels")
    if channels is None:
        channels = {19: MONO19, 16: TUH16}.get(n_channels)
    if channels is None:
        raise ValueError(f"no channel list for {dataset!r} ({n_channels} ch)")
    out = []
    for ch in channels:
        name = str(ch).upper().removeprefix("EEG ").strip()
        if "-" in name:                     # bipolar: midpoint (their rule)
            a, b = name.split("-", 1)
            pa, pb = bank[a.strip()], bank[b.strip()]
            out.append([(x + y) / 2.0 for x, y in zip(pa, pb)])
        else:
            out.append(bank[name])
    return torch.tensor(out, dtype=torch.float32)
