"""Fixed capability evaluator with declared raw-input operator substitution."""
import argparse,hashlib,json
from pathlib import Path
from scripts.experiments import cfc_operator_capabilities as bench
from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator
from paclock_bench.models.paclock.frontend.measurement_uncertain import UncertainPACOperator

parser=argparse.ArgumentParser()
parser.add_argument('--candidate',choices=['baseline','uncertain'],required=True)
parser.add_argument('--block-cycles',type=float,default=2.)
parser.add_argument('--suite',choices=['recovery','grid','null'],required=True)
parser.add_argument('--split',choices=['dev','calibration','confirm'],default='dev')
parser.add_argument('--cases',type=int,default=32)
parser.add_argument('--threads',type=int,default=4)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
bench.SEEDS=dict(dev=24000000,calibration=26000000,confirm=28000000)
if args.candidate=='uncertain':
 class Factory:
  def __new__(cls,*a,**kw):
   if kw.get('harmonics')==1:return PACCurveOperator(*a,**kw)
   return UncertainPACOperator(*a,block_cycles=args.block_cycles,**kw)
 Factory.__module__=UncertainPACOperator.__module__
 bench.PACCurveOperator=Factory
bench.evaluate(args)
p=args.output/'manifest.json';data=json.loads(p.read_text())
data['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
data['split_seeds']=bench.SEEDS
p.write_text(json.dumps(data,indent=2)+'\n')
