from __future__ import annotations

import os, sys, re
import inspect
sys.path.append('./eval')
from runner import *

from typing import Callable, Union

def NormalRpsMajor(config:NormalRpsMajorConfigBase, 
                   model_list, interval_sec, duration, **kargs):
    return MicrobenchmarkInferWorkload_RpsMajor(
        model_list=model_list,
        interval_sec=interval_sec,
        duration=duration,
        rps_dist=config.rps_dist,
        acf=config.acf,
        acf_mse_err=config.acf_mse_err,
        seed=config.seed,
        **kargs)


def NormalModelMajor(config:NormalModelMajorConfigBase,
                     model_list, interval_sec, duration, **kargs):
    return MicrobenchmarkInferWorkload_ModelMajor(
        model_list=model_list,
        interval_sec=interval_sec,
        duration=duration,
        request_model_dist=config.request_model_dist,
        rps_dist=config.rps_dist,
        acf=config.acf,
        acf_mse_err=config.acf_mse_err,
        seed=config.seed,
        **kargs)


def normal_rps_major(name):
    def register_config(config_cls):
        # register workload
        def NormalRpsMajorWrapper(model_list, interval_sec, duration, **kargs):
            return NormalRpsMajor(
                config=config_cls(),
                model_list=model_list,
                interval_sec=interval_sec,
                duration=duration,
                **kargs
            )
        setattr(sys.modules[__name__], name, NormalRpsMajorWrapper)
        return config_cls
    return register_config


def alias_as(name):
    def register_config(config_cls):
        src_name = re.search(r"(.*)Config", config_cls.__name__).group(1)
        exist = getattr(sys.modules[__name__], f'{src_name}Config', None)
        if exist is not None:
            raise ValueError(f"Workload Config {src_name} already exists")
        setattr(sys.modules[__name__], name, getattr(sys.modules[__name__], src_name))
        setattr(sys.modules[__name__], f'{name}Config', config_cls)
        return config_cls
    return register_config


def combine_rps_major_config_by_markov(
    cls_name: str, 
    parents: tuple[type], 
    attr: Dict, 
    config_a: NormalRpsMajorConfigBase,
    config_b: NormalRpsMajorConfigBase, 
    trans_prob: List[float] = [0.3, 0.7],
    acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1), 
    acf_mse_err=0.4
):
    assert parents == (NormalRpsMajorConfigBase,), (
        f"class parents should be ({NormalRpsMajorConfigBase.__name__},), "
        f"but got {parents}")
    assert isinstance(config_a, NormalRpsMajorConfigBase)
    assert isinstance(config_b, NormalRpsMajorConfigBase)
    def init_fn(self):
        seed = re.search(r"(.*)Config", cls_name).group(1)
        seed = get_global_seed_by_hash(seed)
        super(type(self), self).__init__(
            seed, 
            dist.MarkovModulatedDistribution_2(
                    [config_a.rps_dist.dist_param, config_b.rps_dist.dist_param],
                    trans_prob, seed),
            acf, 
            acf_mse_err)
    attr['__init__'] = init_fn
    return type(cls_name, parents, attr)


def combine_model_major_config_by_markov(
    cls_name:str,
    parents: tuple[type],
    attr: dict,
    config_a: NormalModelMajorConfigBase,
    config_b: NormalModelMajorConfigBase,
    trans_prob: List[float] = [0.3, 0.7],
    acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
    acf_mse_err=0.3
):
    assert parents == (NormalModelMajorConfigBase,), \
        f"class parents should be ({NormalModelMajorConfigBase.__name__},), but got {parents}"
    assert isinstance(config_a, NormalModelMajorConfigBase)
    assert isinstance(config_b, NormalModelMajorConfigBase)
    assert config_a.rps_dist.dist_param == config_b.rps_dist.dist_param
    def init_fn(self):
        seed = re.search(r'(.*)Config', cls_name).group(1)
        seed = get_global_seed_by_hash(seed)
        super(type(self), self).__init__(
            seed,
            dist.MarkovModulatedDistribution_2(
                [config_a.request_model_dist.dist_param, config_b.request_model_dist.dist_param],
                trans_prob, seed),
            config_a.rps_dist.dist_param,
            acf,
            acf_mse_err
        )
    attr['__init__'] = init_fn
    return type(cls_name, parents, attr)


def normal_model_major(name):
    def register_config(config_cls):
        # register workload
        def NormalModelMajorWrapper(model_list, interval_sec, duration, **kargs):
            return NormalModelMajor(
                config=config_cls(),
                model_list=model_list,
                interval_sec=interval_sec,
                duration=duration,
                **kargs
            )
        setattr(sys.modules[__name__], name, NormalModelMajorWrapper)
        return config_cls
    return register_config


def get_all_normal_workload(
    name_filter:str=None, 
    base_type_filter:type=None
) -> Dict[str, Callable[[], Union[MicrobenchmarkInferWorkload_RpsMajor, 
                                  MicrobenchmarkInferWorkload_ModelMajor]]]:
    wkld_dict = {}
    for cfg_name, obj in inspect.getmembers(sys.modules[__name__]):
        try:
            if issubclass(obj, NormalRpsMajorConfigBase) or issubclass(obj, NormalModelMajorConfigBase):
                if name_filter is not None and not re.search(name_filter, cfg_name):
                    continue
                if base_type_filter is not None and not issubclass(obj, base_type_filter):
                    continue
                m = re.search(r"(.*)Config$", obj.__name__)
                if m is None:
                    continue
                wkld_name = m.group(1)
                wkld_dict[wkld_name] = getattr(sys.modules[__name__], wkld_name)
        except TypeError:
            pass
    return wkld_dict


class NormalRpsMajorConfigBase:
    def __init__(self, seed, rps_dist, acf, acf_mse_err) -> None:
        self.seed = seed
        if isinstance(rps_dist, dist.DistParamBase):
            self.rps_dist = dist.Distribution(dist_param=rps_dist, seed=self.seed)
        else:
            self.rps_dist = rps_dist
        self.acf = acf
        self.acf_mse_err = acf_mse_err

    def __repr__(self) -> str:
        return f'''{self.__class__.__name__}(
    seed={self.seed}, 
    rps_dist={self.rps_dist}, 
    acf={self.acf}, 
    acf_mse_err={self.acf_mse_err})'''


class NormalModelMajorConfigBase:
    def __init__(self, seed, request_model_dist, rps_dist, acf, acf_mse_err) -> None:
        self.seed = seed
        if isinstance(request_model_dist, dist.DistParamBase):
            self.request_model_dist = dist.Distribution(dist_param=request_model_dist, seed=self.seed)
        else:
            self.request_model_dist = request_model_dist
        if isinstance(rps_dist, dist.DistParamBase):
            self.rps_dist = dist.Distribution(dist_param=rps_dist, seed=self.seed)
        else:
            self.rps_dist = rps_dist
        self.acf = acf
        self.acf_mse_err = acf_mse_err

    def __repr__(self) -> str:
        return f'''{self.__class__.__name__}(
    seed={self.seed}, 
    request_model_dist={self.request_model_dist}, 
    rps_dist={self.rps_dist}, 
    acf={self.acf}, 
    acf_mse_err={self.acf_mse_err})'''


# MARK: 1.1. Normal RPS Major
@normal_rps_major("NormalA_v0")
class NormalA_v0Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA"),
            dist.LogNormalParam(4, 0.5),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_1")
class NormalA_LogNormal_RPS_1Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_1"),
            dist.LogNormalParam(1, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

@normal_rps_major("NormalA_LogNormal_RPS_2")
class NormalA_LogNormal_RPS_2Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_2"),
            dist.LogNormalParam(2, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

@normal_rps_major("NormalA_LogNormal_RPS_3")
class NormalA_LogNormal_RPS_3Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_3"),
            dist.LogNormalParam(3, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_4")
class NormalA_LogNormal_RPS_4Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_4"),
            dist.LogNormalParam(3.5, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_5")
class NormalA_LogNormal_RPS_5Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_5"),
            dist.LogNormalParam(4, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_5")
class NormalA_LogNormal_RPS_5Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_5"),
            dist.LogNormalParam(4, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_6")
class NormalA_LogNormal_RPS_6Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_6"),
            dist.LogNormalParam(4.25, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_7")
class NormalA_LogNormal_RPS_7Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_7"),
            dist.LogNormalParam(4.5, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_8")
class NormalA_LogNormal_RPS_8Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_8"),
            dist.LogNormalParam(4.75, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("NormalA_LogNormal_RPS_9")
class NormalA_LogNormal_RPS_9Config(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("NormalA_LogNormal_RPS_9"),
            dist.LogNormalParam(5, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("Normal_LogNormal_A")
class Normal_LogNormal_AConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_LogNormal_A"),
            dist.LogNormalParam(1, 1),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("Normal_LogNormal_B")
class Normal_LogNormal_BConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_LogNormal_B"),
            dist.LogNormalParam(2.5, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )


@normal_rps_major("Normal_LogNormal_C")
class Normal_LogNormal_CConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_LogNormal_C"),
            dist.LogNormalParam(4.5, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

@normal_rps_major("Normal_LogNormal_D")
class Normal_LogNormal_DConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_LogNormal_D"),
            dist.LogNormalParam(5, 0.3),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

@normal_rps_major("Normal_Weibull_A")
class Normal_Weibull_AConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_Weibull_A"),
            dist.WeibullParam(1, 5),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

@normal_rps_major("Normal_Weibull_B")
class Normal_Weibull_BConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_Weibull_B"),
            dist.WeibullParam(3, 50),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

@normal_rps_major("Normal_Weibull_C")
class Normal_Weibull_CConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_Weibull_C"),
            dist.WeibullParam(3, 100),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

@normal_rps_major("Normal_Weibull_D")
class Normal_Weibull_DConfig(NormalRpsMajorConfigBase):
    def __init__(self) -> None:
        super().__init__(
            get_global_seed_by_hash("Normal_Weibull_D"),
            dist.WeibullParam(3, 150),
            dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            0.4
        )

# MARK: 1.2. +Markov
@normal_rps_major("Normal_Markov_LogNormal_AB")
class Normal_Markov_LogNormal_ABConfig(
        NormalRpsMajorConfigBase, metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_LogNormal_AConfig(), config_b=Normal_LogNormal_BConfig()):
    pass


@normal_rps_major("Normal_Markov_LogNormal_AC")
class Normal_Markov_LogNormal_ACConfig(
        NormalRpsMajorConfigBase, metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_LogNormal_AConfig(), config_b=Normal_LogNormal_CConfig()):
    pass


@normal_rps_major("Normal_Markov_LogNormal_AD")
class Normal_Markov_LogNormal_ADConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_LogNormal_AConfig(), config_b=Normal_LogNormal_DConfig()):
    pass


@normal_rps_major("Normal_Markov_LogNormal_BC")
class Normal_Markov_LogNormal_BCConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_LogNormal_BConfig(), config_b=Normal_LogNormal_CConfig()):
    pass

@normal_rps_major("Normal_Markov_LogNormal_BD")
class Normal_Markov_LogNormal_BDConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_LogNormal_BConfig(), config_b=Normal_LogNormal_DConfig()):
    pass


@normal_rps_major("Normal_Markov_LogNormal_CD")
class Normal_Markov_LogNormal_CDConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_LogNormal_CConfig(), config_b=Normal_LogNormal_DConfig()):
    pass


@normal_rps_major("Normal_Markov_Weibull_AB")
class Normal_Markov_Weibull_ABConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_Weibull_AConfig(), config_b=Normal_Weibull_BConfig()):
    pass


@normal_rps_major("Normal_Markov_Weibull_AC")
class Normal_Markov_Weibull_ACConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_Weibull_AConfig(), config_b=Normal_Weibull_CConfig()):
    pass


@normal_rps_major("Normal_Markov_Weibull_AD")
class Normal_Markov_Weibull_ADConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_Weibull_AConfig(), config_b=Normal_Weibull_DConfig()):
    pass


@normal_rps_major("Normal_Markov_Weibull_BC")
class Normal_Markov_Weibull_BCConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_Weibull_BConfig(), config_b=Normal_Weibull_CConfig()):
    pass

@normal_rps_major("Normal_Markov_Weibull_BD")
class Normal_Markov_Weibull_BDConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_Weibull_BConfig(), config_b=Normal_Weibull_DConfig()):
    pass

@normal_rps_major("Normal_Markov_Weibull_CD")
class Normal_Markov_Weibull_CDConfig(NormalRpsMajorConfigBase,
        metaclass=combine_rps_major_config_by_markov,
        config_a=Normal_Weibull_CConfig(), config_b=Normal_Weibull_DConfig()):
    pass


# @normal_rps_major("NormalA_v1")
# class NormalA_v1Config(NormalRpsMajorConfigBase):
#     def __init__(self) -> None:
#         self.seed = get_global_seed_by_hash("NormalA")
#         # self.rps_dist = dist.Distribution(dist_param=dist.LogNormalParam(4.5, 0.3), 
#         #                                   seed=self.seed)
#         self.rps_dist = dist.Distribution(dist_param=dist.WeibullParam(1, 5), 
#                                           seed=self.seed)
#         # acf = dist.AutoCorrelationGenerator('exp', 10, exp_decay_param=0.15)
#         self.acf = dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1)
#         self.acf_mse_err = 0.4


# @normal_rps_major("NormalA_v2")
# class NormalA_v2Config(NormalRpsMajorConfigBase):
#     def __init__(self) -> None:
#         self.seed = get_global_seed_by_hash("NormalA")
#         self.rps_dist = dist.Distribution(dist_param=dist.LogNormalParam(2, 0.3), 
#                                           seed=self.seed)
#         # acf = dist.AutoCorrelationGenerator('exp', 10, exp_decay_param=0.15)
#         self.acf = dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1)
#         self.acf_mse_err = 0.4


@normal_rps_major("NormalB_v0")
class NormalB_v0Config(NormalRpsMajorConfigBase):
    def __init__(self):
        self.seed = get_global_seed_by_hash("NormalB")
        self.rps_dist =dist.MarkovModulatedDistribution_2(
            dists=[dist.LogNormalParam(2, 0.3), dist.LogNormalParam(4.5, 0.3)],
            trans_prob=[0.3, 0.7],
            seed=self.seed
        ) 
        self.acf = dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1)
        self.acf_mse_err = 0.35


# MARK: 2.1. Normal Model Major
@normal_model_major("Normal_Model_LogNormal_A")
class Normal_Model_LogNormal_AConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_LogNormal_A"),
            # request_model_dist=dist.LogNormalParam(3, 0.5),
            request_model_dist=dist.LogNormalParam(1, 0.5),
            rps_dist=dist.LogNormalParam(4, 0.3),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

@normal_model_major("Normal_Model_LogNormal_B")
class Normal_Model_LogNormal_BConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_LogNormal_B"),
            request_model_dist=dist.LogNormalParam(2, 0.5),
            rps_dist=dist.LogNormalParam(4, 0.3),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

@normal_model_major("Normal_Model_LogNormal_C")
class Normal_Model_LogNormal_CConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_LogNormal_C"),
            request_model_dist=dist.LogNormalParam(2.5, 0.5),
            rps_dist=dist.LogNormalParam(4, 0.3),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

@normal_model_major("Normal_Model_LogNormal_D")
class Normal_Model_LogNormal_DConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_LogNormal_D"),
            request_model_dist=dist.LogNormalParam(3, 0.5),
            rps_dist=dist.LogNormalParam(4, 0.3),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

@normal_model_major("Normal_Model_Weibull_A")
class Normal_Model_Weibull_AConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_Weibull_A"),
            request_model_dist=dist.WeibullParam(1, 5),
            rps_dist=dist.WeibullParam(3, 100),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

@normal_model_major("Normal_Model_Weibull_B")
class Normal_Model_Weibull_BConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_Weibull_B"),
            request_model_dist=dist.WeibullParam(3, 20),
            rps_dist=dist.WeibullParam(3, 100),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

@normal_model_major("Normal_Model_Weibull_C")
class Normal_Model_Weibull_CConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_Weibull_C"),
            request_model_dist=dist.WeibullParam(3, 35),
            rps_dist=dist.WeibullParam(3, 100),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

@normal_model_major("Normal_Model_Weibull_D")
class Normal_Model_Weibull_DConfig(NormalModelMajorConfigBase):
    def __init__(self):
        super().__init__(
            seed=get_global_seed_by_hash("Normal_Model_Weibull_D"),
            request_model_dist=dist.WeibullParam(3, 50),
            rps_dist=dist.WeibullParam(3, 100),
            acf=dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1),
            acf_mse_err=0.3
        )

# MARK: 2.2. +Markov
@normal_model_major("Normal_Model_Markov_LogNormal_AB")
class Normal_Model_Markov_LogNormal_ABConfig(NormalModelMajorConfigBase, 
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_LogNormal_AConfig(), config_b=Normal_Model_LogNormal_BConfig()):
    pass


@normal_model_major("Normal_Model_Markov_LogNormal_AC")
class Normal_Model_Markov_LogNormal_ACConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_LogNormal_AConfig(), config_b=Normal_Model_LogNormal_CConfig()):
    pass


@normal_model_major("Normal_Model_Markov_LogNormal_AD")
class Normal_Model_Markov_LogNormal_ADConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_LogNormal_AConfig(), config_b=Normal_Model_LogNormal_DConfig()):
    pass


@normal_model_major("Normal_Model_Markov_LogNormal_BC")
class Normal_Model_Markov_LogNormal_BCConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_LogNormal_BConfig(), config_b=Normal_Model_LogNormal_CConfig()):
    pass


@normal_model_major("Normal_Model_Markov_LogNormal_BD")
class Normal_Model_Markov_LogNormal_BDConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_LogNormal_BConfig(), config_b=Normal_Model_LogNormal_DConfig()):
    pass


@normal_model_major("Normal_Model_Markov_LogNormal_CD")
class Normal_Model_Markov_LogNormal_CDConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_LogNormal_CConfig(), config_b=Normal_Model_LogNormal_DConfig()):
    pass


@normal_model_major("Normal_Model_Markov_Weibull_AB")
class Normal_Model_Markov_Weibull_ABConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_Weibull_AConfig(), config_b=Normal_Model_Weibull_BConfig()):
    pass


@normal_model_major("Normal_Model_Markov_Weibull_AC")
class Normal_Model_Markov_Weibull_ACConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_Weibull_AConfig(), config_b=Normal_Model_Weibull_CConfig()):
    pass


@normal_model_major("Normal_Model_Markov_Weibull_AD")
class Normal_Model_Markov_Weibull_ADConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_Weibull_AConfig(), config_b=Normal_Model_Weibull_DConfig()):
    pass


@normal_model_major("Normal_Model_Markov_Weibull_BC")
class Normal_Model_Markov_Weibull_BCConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_Weibull_BConfig(), config_b=Normal_Model_Weibull_CConfig()):
    pass


@normal_model_major("Normal_Model_Markov_Weibull_BD")
class Normal_Model_Markov_Weibull_BDConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_Weibull_BConfig(), config_b=Normal_Model_Weibull_DConfig()):
    pass


@normal_model_major("Normal_Model_Markov_Weibull_CD")
class Normal_Model_Markov_Weibull_CDConfig(NormalModelMajorConfigBase,
        metaclass=combine_model_major_config_by_markov,
        config_a=Normal_Model_Weibull_CConfig(), config_b=Normal_Model_Weibull_DConfig()):
    pass


@normal_model_major("NormalC_v0")
class NormalC_v0Config:
    def __init__(self) -> None:
        self.seed = get_global_seed_by_hash("NormalC")
        # self.request_model_dist = dist.Distribution(dist.LogNormalParam(2, 1), seed=self.seed)
        self.request_model_dist = dist.Distribution(dist.GammaParam(2, 2), seed=self.seed)
        self.rps_dist = dist.Distribution(dist.LogNormalParam(4, 0.5), seed=self.seed)
        self.acf = dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1)
        self.acf_mse_err = 0.35


@normal_model_major("NormalD")
class NormalDConfig:
    def __init__(self) -> None:
        self.seed = get_global_seed_by_hash("NormalD")
        self.request_model_dist = dist.MarkovModulatedDistribution_2(
            dists=[dist.LogNormalParam(2, 0.3), dist.LogNormalParam(4.5, 0.3)],
            trans_prob=[0.3, 0.7],
            seed=self.seed
        )
        self.rps_dist = dist.Distribution(dist.LogNormalParam(4, 0.5), seed=self.seed)
        self.acf = dist.AutoCorrelationGenerator('inverse', 5, inverse_decay_param=1)
        self.acf_mse_err = 0.35
    

# def NormalA(model_list, interval_sec, duration, **kargs):
#     # normal_a_config = NormalAConfig()
#     normal_a_config = NormalBConfig()

#     return MicrobenchmarkInferWorkload_RpsMajor(
#         model_list=model_list,
#         interval_sec=interval_sec,
#         duration=duration,
#         rps_dist=normal_a_config.rps_dist,
#         acf=normal_a_config.acf,
#         acf_mse_err=normal_a_config.acf_mse_err,
#         seed=normal_a_config.seed,
#         **kargs
#     )
