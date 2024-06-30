import os, sys
import abc
import numpy as np
from dataclasses import dataclass
from numpy.random import MT19937, SeedSequence
import math

class DistParamBase:
    pass

@dataclass
class NormalParam(DistParamBase):
    mean: float
    sigma: float

@dataclass
class LogNormalParam(DistParamBase):
    mean: float
    sigma: float

@dataclass
class WeibullParam(DistParamBase):
    shape: float
    scale: float

@dataclass
class GammaParam(DistParamBase):
    shape: float
    scale: float

@dataclass
class GEVParam(DistParamBase):
    shape: float
    loc: float
    scale: float

@dataclass
class ZipfParam(DistParamBase):
    alpha: float


class DistributionBase(abc.ABC):
    @abc.abstractmethod
    def get(self):
        pass

class Distribution(DistributionBase):
    dist_fn_name_dict = {
        NormalParam: 'normal',
        LogNormalParam: 'lognormal',
        WeibullParam: 'weibull',
        GEVParam: 'gev',
        ZipfParam: 'zipf'
    }

    def __init__(self, dist_param, seed):
        super().__init__()
        self.dist_param = dist_param
        self.seed = seed
        # print(dist_param, seed)
        self.rs = np.random.RandomState(MT19937(SeedSequence(seed)))

        self.dist_fn = None
        for param_type, dist_fn_name in self.dist_fn_name_dict.items():
            if isinstance(dist_param, param_type):
                self.dist_fn = getattr(self.rs, dist_fn_name)
                break
        if self.dist_fn is None:
            raise ValueError(f'Invalid distribution parameter type: {type(dist_param)}')
        
    def __repr__(self) -> str:
        return f'[{self.dist_param} seed={self.seed}]'

    def get(self):
        if isinstance(self.dist_param, WeibullParam):
            # accroding to numpy, we have to manually set the scale parameter
            return self.dist_param.scale * self.dist_fn(self.dist_param.shape)
        else:
            return self.dist_fn(**self.dist_param.__dict__)


class MarkovModulatedDistribution_2(DistributionBase):
    def __init__(self,
                 dists: list[Distribution | DistParamBase],
                 trans_prob: list[float],
                 seed):
        super().__init__()
        if not len(dists) == 2 or not len(trans_prob) == 2:
            raise Exception('MoMarkovModulatedDistribution_2 requires 2 distributions')

        trans_prob = np.array(trans_prob)
        if not np.all((trans_prob >= 0) & (trans_prob <= 1)):
            raise Exception('Invalid transition probability')

        for i in range(len(dists)):
            if isinstance(dists[i], DistParamBase):
                dists[i] = Distribution(dists[i], seed)

        self.dists = dists
        self.trans_mat = np.array([
            # diag: keep same dist
            [1-trans_prob[0], trans_prob[0]],
            [trans_prob[1], 1-trans_prob[1]] 
        ])
        self.seed = seed
        self.rs = np.random.RandomState(MT19937(SeedSequence(seed)))
        self.state = 0
    
    def __repr__(self) -> str:
        return f'Markov({self.dists[0]} {self.trans_mat[0]}, {self.dists[1]} {self.trans_mat[1]} seed={self.seed})'

    def get(self):
        dist = self.dists[self.state]
        res = dist.get()

        rand = self.rs.rand()
        if rand > self.trans_mat[self.state][self.state]:
            self.state = 1 - self.state

        return res
    

class AutoCorrelationGenerator:
    def __init__(self, decay_type, period,
                 exp_decay_param = None,
                 inverse_decay_param = None):
        self.decay_type = decay_type
        self.exp_decay_param = exp_decay_param
        self.inverse_decay_param = inverse_decay_param
        self.period = period
        if decay_type == 'exp':
            self.decay_fn = self.exp_decay_fn(exp_decay_param, period)
        elif decay_type == 'inverse':
            self.decay_fn = self.inverse_decay_fn(inverse_decay_param, period)
        else:
            raise ValueError('Invalid decay type')
        
    def __repr__(self) -> str:
        if self.decay_type == 'exp':
            return f'ACF([exp {self.exp_decay_param}] period={self.period})'
        elif self.decay_type == 'inverse':
            return f'ACF([inverse {self.inverse_decay_param}] period={self.period})'
        else:
            raise ValueError('Invalid decay type')

    def exp_decay_fn(self, exp_param, t):
        def fn(x):
            return math.exp(-exp_param * x) * math.cos(2 * math.pi * x / t)
        return fn
    
    def inverse_decay_fn(self, order, t):
        def fn(x):
            return 1 / (1 + x ** order) * math.cos(2 * math.pi * x / t)
        return fn
    
    def get(self, x):
        return self.decay_fn(x)


def normalized_acf(x):
    from scipy.signal import correlate

    mean = np.mean(x)
    var = np.var(x)
    acf = correlate(x - mean, x - mean, mode='full')
    acf /= len(x) * var
    return acf[len(acf)//2:]

