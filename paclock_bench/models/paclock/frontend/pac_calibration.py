"""Independent-record PAC score calibration, separate from token features.

Rank guarantees require exchangeable reference/test null records. Matching an
input signature is necessary, but does not establish that distributional
assumption for real EEG. A physical-valid flag is never a significance flag.
"""
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch


def calibration_rank(n: int, alpha: float, confidence: float | None = None) -> int:
    """One-based order statistic; n+1 means an always-abstaining threshold.

    None: split-rank rule with marginal FPR <= alpha under exchangeability.
    confidence: binomial tolerance rule, P(population FPR<=alpha)>=confidence
    under iid calibration scores. Strict score>threshold handles ties safely.
    """
    if n<1 or not 0<alpha<1 or (confidence is not None and not 0<confidence<1):
        raise ValueError('invalid calibration size or probabilities')
    if confidence is None:
        return min(n+1,math.ceil((n+1)*(1-alpha)))
    p=1-alpha
    cumulative=0.
    for successes in range(n+1):
        logp=(math.lgamma(n+1)-math.lgamma(successes+1)-math.lgamma(n-successes+1)
              +successes*math.log(p)+(n-successes)*math.log1p(-p))
        cumulative+=math.exp(logp)
        if cumulative>=confidence:
            return successes+1
    return n+1


def measurement_signature(operator, channels: int, samples: int) -> str:
    if channels<1 or samples<operator.analysis.patch_len:
        raise ValueError('positive channels and at least one token patch are required')
    from .measurement_curve import PACCurveOperator,curve_regression
    from .measurement import linear_analytic_filter
    source=hashlib.sha256()
    for function in (type(operator)._forward,PACCurveOperator._forward,
                     PACCurveOperator._source_analysis,curve_regression,linear_analytic_filter):
        source.update(inspect.getsource(function).encode())
    state=hashlib.sha256()
    for name,value in sorted(operator.state_dict().items()):
        if isinstance(value,torch.Tensor):
            state.update(name.encode());state.update(str(value.dtype).encode())
            state.update(str(tuple(value.shape)).encode())
            state.update(value.detach().cpu().contiguous().numpy().tobytes())
    data=dict(operator=type(operator).__name__,configuration=operator.configuration,
              score_source_sha256=source.hexdigest(),physical_state_sha256=state.hexdigest(),
              torch_version=torch.__version__,device_type=operator.analysis.centers.device.type,
              channels=channels,samples=samples,score='max_predictive_gain_over_channels_and_pairs')
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def record_scores(measurement):
    scores=measurement.predictive_gain
    if scores.ndim!=4:
        raise ValueError('expected predictive_gain [batch,channel,source,target]')
    return scores.flatten(1).amax(-1)


@dataclass(frozen=True)
class PACCalibration:
    threshold: float | None
    rank: int
    n: int
    alpha: float
    confidence: float | None
    signature: str
    reference_sha256: str

    @classmethod
    def fit(cls,scores,operator,channels: int,samples: int,alpha: float=.05,
            confidence: float | None=None):
        values=torch.as_tensor(scores,dtype=torch.float64).detach().cpu()
        if values.ndim!=1 or not len(values) or not bool(torch.isfinite(values).all()):
            raise ValueError('one finite maximum score per independent null record is required')
        k=calibration_rank(len(values),alpha,confidence)
        threshold=float(values.sort().values[k-1]) if k<=len(values) else None
        return cls(threshold,k,len(values),alpha,confidence,
                   measurement_signature(operator,channels,samples),
                   hashlib.sha256(values.numpy().tobytes()).hexdigest())

    def detect(self,measurement,operator,channels: int,samples: int):
        if self.signature!=measurement_signature(operator,channels,samples):
            raise ValueError('calibration does not match operator, channels, or record length')
        if measurement.predictive_gain.shape[1]!=channels:
            raise ValueError('declared channel count does not match measurement')
        if measurement.source_phase.shape[-1]!=samples:
            raise ValueError('declared record length does not match measurement')
        if measurement.predictive_gain.shape[-2:]!=(operator.n_source,operator.n_target):
            raise ValueError('frequency axes do not match operator')
        score=record_scores(measurement)
        return torch.zeros_like(score,dtype=torch.bool) if self.threshold is None else score>self.threshold

    def save(self,path):
        Path(path).write_text(json.dumps(asdict(self),indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        return cls(**json.loads(Path(path).read_text()))
