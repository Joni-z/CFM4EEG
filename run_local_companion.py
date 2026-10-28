import argparse
from pathlib import Path
from types import SimpleNamespace
from scripts.experiments import cfc_operator_capabilities as b
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
b.SEEDS['confirm']=36000000
b.evaluate(SimpleNamespace(output=a.output,suite='local',split='confirm',cases=64,threads=3))
