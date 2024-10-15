from .normal import *

from .skew import construct_skew_configs
construct_skew_configs(1.05)
from .skew import *


def _set_workload_alias(workload_name_alias: Dict[str, str]):
    for dst_name, src_name in workload_name_alias.items():
        src_config_cls_name = f'{src_name}Config'
        src_config_cls = getattr(sys.modules[__name__], src_config_cls_name)
        src_wkld_type = getattr(sys.modules[__name__], src_name)

        dst_config_cls_name = f'{dst_name}Config'
        assert not hasattr(sys.modules[__name__], dst_name)
        assert not hasattr(sys.modules[__name__], dst_config_cls_name)

        setattr(sys.modules[__name__], dst_name, src_wkld_type)
        setattr(sys.modules[__name__], dst_config_cls_name, src_config_cls)


_set_workload_alias({
    # 'NormalA' : 'Normal_Markov_LogNormal_AC',
    # 'NormalB' : 'Normal_Model_Markov_LogNormal_AD',
    # 'NormalC' : 'Normal_Model_LogNormal_D',

    # 'SkewA'   : 'Skew_Markov_LogNormal_AC',
    # 'SkewB'   : 'Skew_Model_Markov_LogNormal_AD',
    # 'SkewC'   : 'Skew_Model_LogNormal_D'

    'NormalA' : 'Normal_LogNormal_A',
    'NormalB' : 'Normal_LogNormal_C',
    'NormalC' : 'Normal_Markov_LogNormal_AC',

    'SkewA'   : 'Skew_LogNormal_A',
    'SkewB'   : 'Skew_LogNormal_C',
    'SkewC'   : 'Skew_Markov_LogNormal_AC',
})