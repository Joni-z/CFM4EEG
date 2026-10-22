"""Restore a selected original-author repository at its recorded revision."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sources = json.loads((ROOT / 'upstream/sources.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?', choices=sorted(sources))
    args = parser.parse_args()
    if args.model is None:
        for name, source in sources.items():
            print(name, source['commit'], source['url'])
        return
    source = sources[args.model]
    target = ROOT / 'vendor' / source['directory']
    if target.exists():
        head = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(target), 'status', '--porcelain'], text=True)
        if head != source['commit'] or dirty:
            raise RuntimeError(f'Existing checkout differs; leave it untouched: {target}')
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', source['url'], str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'fetch', '--depth=1', 'origin', source['commit']], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', '--detach', source['commit']], check=True)
    for filename in source['entrypoints']:
        if not (target / filename).is_file():
            raise FileNotFoundError(target / filename)
    print(f'{args.model}: {source["commit"]} at {target}')


if __name__ == '__main__':
    main()
