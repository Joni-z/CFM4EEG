import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('watch',Path(__file__).with_name('watch.py'));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
def row(tag,pr,auc=.6):return dict(tag=tag,train_loss=.01,val_loss=.02,metrics=dict(pr_auc=pr,auroc=auc,balanced_acc=.5))
def flat(epoch,pr):return {'history':[row(f'epoch {epoch} step {i*200} |',pr) for i in range(1,40)]+[row(f'epoch {epoch} |',pr)],'active':True}
class Gates(unittest.TestCase):
 def test_first_epoch_not_killed(self):
  for p in [.006,.0291,.0546,.0716,.0739,.1235]:self.assertIsNone(w.decision(w.NAMES[0],flat(0,p)))
 def test_epoch2_severe_plateau(self):self.assertEqual(w.decision(w.NAMES[0],flat(1,.08)),'epoch2_low_plateau')
 def test_epoch3_moderate_plateau(self):self.assertEqual(w.decision(w.NAMES[0],flat(2,.24)),'epoch3_low_plateau')
 def test_rising_not_killed(self):
  r=flat(2,.2);r['history'][-1]['metrics']['pr_auc']=.25;self.assertIsNone(w.decision(w.NAMES[0],r))
 def test_historical_epoch2_range_retained(self):
  for v in [.4312,.4493,.4631,.4808]:self.assertIsNone(w.decision(w.NAMES[0],flat(1,v)))
 def test_missing_data_is_not_failure(self):self.assertIsNone(w.decision(w.NAMES[0],{'read_error':'partial file'}))
 def test_nonfinite(self):
  r=flat(0,.01);r['history'][-1]['train_loss']=float('nan');self.assertEqual(w.decision(w.NAMES[0],r),'nonfinite')
 def test_actual_tuev_best_passes(self):
  r={'history':[{'tag':'epoch 8 |','metrics':{'cohen_kappa':.66165,'balanced_acc':.5437}}],'result':{'stopped_by':'epochs'}}
  self.assertIsNone(w.decision('tuev-nb16_quadrature_20260915',r))
 def test_unfinished_tuev_not_killed(self):
  r={'history':[{'tag':'epoch 10 |','metrics':{'cohen_kappa':.55,'balanced_acc':.52}}]};self.assertIsNone(w.decision('tuev-x',r))
 def test_no_duplicate_submissions_across_cycles(self):
  r=flat(3,.1);r.update(active=False,result={'stopped_by':'epochs'})
  amd={'runs':{w.NAMES[0]:r},'smoke':{}};torch={'queue':'','legacy':{}}
  with tempfile.TemporaryDirectory() as td,patch.object(w,'ROOT',Path(td)),patch.object(w,'ssh',side_effect=lambda host,code:amd if host=='amd' else torch),patch.object(w,'smoke_ready',return_value=True),patch.object(w,'submit',return_value={'state':'submitted','job_id':'123'}) as submit:
   w.main();w.main();self.assertEqual(submit.call_count,2)
 def test_capacity_limit(self):
  r=flat(3,.1);r.update(active=False,result={'stopped_by':'epochs'})
  amd={'runs':{w.NAMES[0]:r,**{n:flat(0,.05) for n in w.NAMES[1:]}},'smoke':{}};torch={'queue':'','legacy':{}}
  with tempfile.TemporaryDirectory() as td,patch.object(w,'ROOT',Path(td)),patch.object(w,'ssh',side_effect=lambda host,code:amd if host=='amd' else torch),patch.object(w,'smoke_ready',return_value=True),patch.object(w,'submit',return_value={'state':'submitted','job_id':'123'}) as submit:
   w.main();self.assertEqual(submit.call_count,1)
if __name__=='__main__':unittest.main()
