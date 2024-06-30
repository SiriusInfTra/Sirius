from runner import *
import runner.distribution as dist
from . import normal
import typing
import inspect


def get_model_hotness(zipf_alpha:float, num_models:int, 
                      seed:Optional[int]=None, sample_size:int=10000):
    if seed is None:
        # we can get same hotness for different workloads
        seed = get_global_seed_by_hash("model_hotness")
    rs = np.random.RandomState(MT19937(SeedSequence(seed)))
    zipf_seq = rs.zipf(zipf_alpha, sample_size)
    zipf_freq = np.zeros(1 + num_models)
    for i in zipf_seq:
        if i <= num_models:
            zipf_freq[i] += 1
    zipf_freq = zipf_freq[1:]
    zipf_freq = zipf_freq / sum(zipf_freq)
    zipf_freq = rs.permutation(zipf_freq)
    return zipf_freq


class SkewRpsMajorConfigBase:
    def __init__(self, zipf_alpha, seed, rps_dist, acf, acf_mse_err) -> None:
        self.zipf_alpha = zipf_alpha
        self.seed = seed
        if isinstance(rps_dist, dist.DistParamBase):
            self.rps_dist = dist.Distribution(dist_param=rps_dist, seed=seed)
        else:
            self.rps_dist = rps_dist
        self.acf = acf
        self.acf_mse_err = acf_mse_err

    def __repr__(self) -> str:
        return f'''{self.__class__.__name__}(
    zipf_alpha={self.zipf_alpha},
    seed={self.seed},
    rps_dist={self.rps_dist},
    acf={self.acf})'''
    

class SkewModelMajorConfigBase:
    def __init__(self, zipf_alpha, seed, request_model_dist, rps_dist, acf, acf_mse_err) -> None:
        self.zipf_alpha = zipf_alpha
        self.seed = seed
        if isinstance(request_model_dist, dist.DistParamBase):
            self.request_model_dist = dist.Distribution(dist_param=request_model_dist, seed=seed)
        else:
            self.request_model_dist = request_model_dist
        if isinstance(rps_dist, dist.DistParamBase):
            self.rps_dist = dist.Distribution(dist_param=rps_dist, seed=seed)
        else:
            self.rps_dist = rps_dist
        self.acf = acf
        self.acf_mse_err = acf_mse_err

    def __repr__(self):
        return f'''{self.__class__.__name__}(
    zipf_alpha={self.zipf_alpha},
    seed={self.seed},
    request_model_dist={self.request_model_dist},
    rps_dist={self.rps_dist},
    acf={self.acf},
    acf_mse_err={self.acf_mse_err})'''


def SkewRpsMajor(config:SkewRpsMajorConfigBase,
                 model_list, interval_sec, duration, **kargs):
    return MicrobenchmarkInferWorkload_RpsMajor(
        model_list=model_list,
        interval_sec=interval_sec,
        duration=duration,
        rps_dist=config.rps_dist,
        acf=config.acf,
        acf_mse_err=config.acf_mse_err,
        seed=config.seed,
        model_hotness=get_model_hotness(config.zipf_alpha, len(model_list)),
        **kargs
    )

def SkewModelMajor(config:SkewModelMajorConfigBase,
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
        model_hotness=get_model_hotness(config.zipf_alpha, len(model_list)),
        **kargs
    )

def skew_rps_major(name):
    def register_config(config_cls):
        def SkewRpsMajorWrapper(model_list, interval_sec, duration, **kargs):
            return SkewRpsMajor(config_cls(), model_list, interval_sec, duration, **kargs)
        setattr(sys.modules[__name__], name, SkewRpsMajorWrapper)
        return config_cls
    return register_config


def skew_model_major(name):
    def register_config(config_cls):
        def SkewModelMajorWrapper(model_list, interval_sec, duration, **kargs):
            return SkewModelMajor(config_cls(), model_list, interval_sec, duration, **kargs)
        setattr(sys.modules[__name__], name, SkewModelMajorWrapper)
        return config_cls
    return register_config


def get_all_skew_workload(name_filter:str=None, 
                            base_type_filter:type=None) -> Dict:
    wkld_dict = {}
    for cfg_name, obj in inspect.getmembers(sys.modules[__name__]):
        try:
            if issubclass(obj, SkewRpsMajorConfigBase) or issubclass(obj, SkewModelMajorConfigBase):
                if name_filter is not None and not re.search(name_filter, cfg_name):
                    continue
                if base_type_filter is not None and not issubclass(obj, base_type_filter):
                    continue                    
                wkld_name = re.search(r"(.*)Config", obj.__name__).group(1)
                wkld_dict[wkld_name] = getattr(sys.modules[__name__], wkld_name)
        except TypeError:
            pass
    return wkld_dict


def construct_skew_config_from_normal(
        cls_name: str, 
        parents: type, 
        attr: Dict,
        zipf_alpha,
        normal_config: normal.NormalRpsMajorConfigBase | normal.NormalModelMajorConfigBase):
    if isinstance(normal_config, normal.NormalRpsMajorConfigBase):
        assert parents == (SkewRpsMajorConfigBase, )
        def init_fn(self):
            super(type(self), self).__init__(
                zipf_alpha=zipf_alpha,
                seed=normal_config.seed,
                rps_dist=normal_config.rps_dist,
                acf=normal_config.acf,
                acf_mse_err=normal_config.acf_mse_err
            )
        attr['__init__'] = init_fn
        skew_config_cls = type(cls_name, parents, attr)
        wkld_type = re.search(r'(.*)Config', cls_name).group(1)
        skew_rps_major(wkld_type)(skew_config_cls)
        return skew_config_cls
    elif isinstance(normal_config, normal.NormalModelMajorConfigBase):
        assert parents == (SkewRpsMajorConfigBase, )
        def init_fn(self):
            super(type(self), self).__init__(
                zipf_alpha=zipf_alpha,
                seed=normal_config.seed,
                request_model_dist=normal_config.request_model_dist,
                rps_dist=normal_config.rps_dist,
                acf=normal_config.acf,
                acf_mse_err=normal_config.acf_mse_err
            )
        attr['__init__'] = init_fn
        skew_config_cls = type(cls_name, parents, attr)
        wkld_type = re.search(r'(.*)Config', cls_name).group(1)
        skew_model_major(wkld_type)(skew_config_cls)
        return skew_config_cls
    else:
        raise ValueError(f'Invalid normal_config type: {type(normal_config)}')
    


# construct skew config from normal config
def construct_skew_configs(zipf_alpha:float):
    for object in inspect.getmembers(normal):
        try:
            if not issubclass(object, (normal.NormalRpsMajorConfigBase, normal.NormalModelMajorConfigBase)):
                continue
            if issubclass(object, normal.NormalRpsMajorConfigBase):
                parents = (SkewRpsMajorConfigBase, )
            else:
                parents = (SkewModelMajorConfigBase, )

            cls_name = object.__name__.replace('Normal', 'Skew')
            skew_config_cls = construct_skew_config_from_normal(
                cls_name, parents, {}, 
                zipf_alpha=zipf_alpha, 
                normal_config=object())
            setattr(sys.modules[__name__], cls_name, skew_config_cls)
        except TypeError:
            pass
