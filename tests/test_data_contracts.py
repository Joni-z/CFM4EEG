import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from paclock_bench.data.contracts import labram_divisor, verify_arrays, resolve_signal_transform, describe_contract

class DataContractTests(unittest.TestCase):
    def test_labram_no_double_scaling(self):
        m={'protocol':{'normalization':'div100'}}
        self.assertEqual(labram_divisor({},m),1)
        with self.assertRaises(ValueError):labram_divisor({'loader_divisor':100},m)
        self.assertEqual(labram_divisor({}, {'protocol':{'units':'uV'}}),100)
        with self.assertRaises(ValueError):labram_divisor({}, {'protocol':{}})
    def test_payload_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            x=Path(d)/'x.npy';y=Path(d)/'y.npy'
            np.save(x,np.zeros((3,2,10),dtype=np.float32));np.save(y,np.zeros(3,dtype=np.int64))
            verify_arrays(np.load(x,mmap_mode='r'),np.load(y),x,y)
            with open(x,'ab')as f:f.write(b'extra')
            with self.assertRaises(ValueError):verify_arrays(np.load(x,mmap_mode='r'),np.load(y),x,y)
    def test_q95_sidecar_recovers_physical_scale(self):
        rng=np.random.default_rng(2);raw=rng.normal(0,20,(8,16,200)).astype(np.float32)
        denominator=np.quantile(abs(raw),.95,axis=-1,keepdims=True)+1e-8
        normalized=(raw/denominator).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            np.save(Path(d)/'train_q95_denominators.npy',denominator.astype(np.float64))
            np.save(Path(d)/'train_signals.npy',normalized)
            scales,multiplier=resolve_signal_transform(d,'train',{'loader_signal_mode':'uv_div100'},{'protocol':{'normalization':'q95'}})
            np.testing.assert_allclose(normalized*scales*multiplier,raw/100,rtol=2e-7,atol=1e-8)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):resolve_signal_transform(d,'train',{'loader_signal_mode':'uv_div100'},{'protocol':{'normalization':'q95'}})
    def test_wrong_hz_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            m={'protocol':{'sample_rate':100}};(Path(d)/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):describe_contract({'sample_rate':200},m,d,'shared')
if __name__=='__main__':unittest.main()
