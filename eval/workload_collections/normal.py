import os, sys
sys.path.append('./eval')
from runner import *


class NormalAConfig:
    def __init__(self) -> None:
        self.seed = get_global_seed_by_hash("NormalA")
        self.rps_dist = dist.Distribution(dist_param=dist.LogNormalParam(4, 0.5), seed=self.seed)
        # acf = dist.AutoCorrelationGenerator('exp', 10, exp_decay_param=0.15)
        self.acf = dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1)
        self.acf_mse_err = 0.4


def NormalA(model_list, interval_sec, duration, **kargs):
    normal_a_config = NormalAConfig()

    return MicrobenchmarkInferWorkload_RpsMajor(
        model_list=model_list,
        interval_sec=interval_sec,
        duration=duration,
        rps_dist=normal_a_config.rps_dist,
        acf=normal_a_config.acf,
        acf_mse_err=normal_a_config.acf_mse_err,
        seed=normal_a_config.seed,
        **kargs
    )