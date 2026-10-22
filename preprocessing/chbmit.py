"""CHB-MIT annotation/channel readers.

Dataset generation uses preprocessing.rebuild_clinical, which applies training-only
augmentation, natural validation/test windows, and grouped patient splitting.
The previous standalone generator is preserved outside the active source tree.
"""
from __future__ import annotations
import re
import mne
import numpy as np

def parse_summary(path: str) -> dict[str, list[tuple[float, float]]]:
    """chbXX-summary.txt -> {edf_filename: [(start_sec, end_sec), ...]}.

    Handles both layouts in the corpus: the common
    ``Seizure Start Time: N seconds`` and chb24's numbered
    ``Seizure 1 Start Time: N seconds``.
    """
    out: dict[str, list[tuple[float, float]]] = {}
    cur = None
    starts: list[float] = []
    ends: list[float] = []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                if cur is not None:
                    out[cur] = list(zip(starts, ends))
                cur = line.split(":", 1)[1].strip()
                starts, ends = [], []
            elif "Start Time:" in line and "Seizure" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    starts.append(float(m.group(1)))
            elif "End Time:" in line and "Seizure" in line:
                m = re.search(r"(\d+)\s*seconds", line)
                if m:
                    ends.append(float(m.group(1)))
    if cur is not None:
        out[cur] = list(zip(starts, ends))
    return out

def pick_channels(raw: mne.io.BaseRaw, wanted: list[str]) -> np.ndarray:
    """CHB-MIT EDFs are already bipolar. Select the 16 named channels in order.

    Most CHB-MIT files list 'T8-P8' twice; MNE de-duplicates by appending an
    index, so the channels arrive as 'T8-P8-0' and 'T8-P8-1'. Matching only on
    the exact name would drop the montage entirely, so a wanted name also
    accepts a '<name>-<digit>' variant, lowest index first (the two copies are
    identical signals).
    """
    def norm(s: str) -> str:
        return s.upper().replace(" ", "")

    index: dict[str, int] = {}
    for i, name in enumerate(raw.ch_names):
        index.setdefault(norm(name), i)

    picked, missing = [], []
    for c in wanted:
        key = norm(c)
        if key in index:
            picked.append(index[key])
            continue
        variants = sorted(k for k in index
                          if k.startswith(key + "-") and k[len(key) + 1:].isdigit())
        if variants:
            picked.append(index[variants[0]])
        else:
            missing.append(c)
    if missing:
        raise KeyError(f"missing channels: {missing}")
    data = raw.get_data()
    return np.stack([data[i] for i in picked])

if __name__ == "__main__":
    raise SystemExit("Use python -m preprocessing.rebuild_clinical --dataset chbmit --help")
