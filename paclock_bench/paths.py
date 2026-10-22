"""Data, weights and author checkouts; no cluster-specific fallback paths."""
from __future__ import annotations
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get('PACLOCK_DATA', os.path.join(REPO, 'data', 'raw'))
PROC_ROOT = os.environ.get('PACLOCK_PROC', os.path.join(REPO, 'data'))
CKPT = os.environ.get('PACLOCK_CKPT', os.path.join(REPO, 'ckpt'))
VENDOR = os.path.join(REPO, 'vendor')
RUNS = os.path.join(REPO, 'runs')
RESULTS = os.path.join(REPO, 'results')


def processed(protocol: str = 'processed', dataset: str = '') -> str:
    return os.path.join(PROC_ROOT, protocol, dataset) if dataset else os.path.join(PROC_ROOT, protocol)


def vendored(name: str) -> str:
    return os.path.join(VENDOR, name)


def expand(path: str) -> str:
    if not path:
        return path
    os.environ.setdefault('PACLOCK_DATA', DATA)
    os.environ.setdefault('PACLOCK_PROC', PROC_ROOT)
    os.environ.setdefault('PACLOCK_CKPT', CKPT)
    return os.path.expanduser(os.path.expandvars(path))
