import enum
import pandas as pd
import pathlib
import os, sys, re
import subprocess
import numpy as np
from typing import List, Union

__parser_dir_path__ = pathlib.Path(__file__).parent.absolute()
__plot_script_dir_path__ = __parser_dir_path__.parent / 'plot_scripts'


class TestUnit(enum.Enum):
    OVER_ALL_SINGLE_GPU = 'over_all_single_gpu'
    OVER_ALL_MULTI_GPU = 'over_all_multi_gpu'
    BREAKDOWN_SINGLE_GPU = 'breakdown_single_gpu'
    BREAKDOWN_MULTI_GPU = 'breakdown_multi_gpu'
    ABLATION = 'ablation'
    UNBALANCE = 'unbalance'
    MEMORY_PRESSURE = 'memory_pressure'
    LLM = 'llm'


def get_plot_script_path(name: str):
    path = __plot_script_dir_path__ / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Plot script {name} not found in {path}")
    return path


def static_partition_to_number_name(name: str):
    if name == 'SP-I':
        return 'SP-75'
    elif name == 'SP-F':
        return 'SP-50'
    else:
        return name   


# MARK: Log Parse Functions
def _parse_system_and_trace_helper(log: Union[str, pathlib.Path]):
    if type(log) != pathlib.Path:
        log = pathlib.Path(log) 
    assert log.exists(), f"Log file {log} does not exist"
    assert log.is_dir(), f"Log file {log} is not a directory"
    log_dir_name = log.name
    system, trace = None, None

    # system name
    if 'colsys' in log_dir_name:
        system = 'ColSys'
    elif 'infer-only' in log_dir_name:
        system = 'Infer-Only'
    elif 'task-switch' in log_dir_name:
        system = 'TaskSwitch'
    elif 'static-partition' in log_dir_name:
        if re.search(r'static-partition-([a-zA-Z_]+)-I', log_dir_name) is not None:
            system = 'SP-I'
        elif re.search(r'static-partition-I', log_dir_name) is not None:
            system = 'SP-I'
        elif re.search(r'static-partition-([a-zA-Z_]+)-F', log_dir_name) is not None:
            system = 'SP-F'
        elif re.search(r'static-partition-F', log_dir_name) is not None:
            system = 'SP-F'
    elif 'um-mps' in log_dir_name:
        system = 'UM+MPS'
    
    # trace name
    if 'NormalA' in log_dir_name or 'NormalLight' in log_dir_name:
        trace = 'Light'
    elif ('NormalB' in log_dir_name or 'NormalHeavy' in log_dir_name) \
        and 'NormalBurst' not in log_dir_name:
        trace = 'Heavy'
    elif 'NormalC' in log_dir_name or 'NormalBurst' in log_dir_name:
        trace = 'Burst'
    elif 'SkewC' in log_dir_name or 'SkewBurst' in log_dir_name:
        trace = 'Skewed'
    elif 'azure' in log_dir_name or 'azure' in log.parent.name:
        trace = 'MAF'

    print(system, trace)
    return system, trace


def _parse_system_and_trace(unit: TestUnit, log: pathlib.Path):
    log_str = str(log)
    def _parse_num_gpu(log_str):
        m = re.search(r'([0-9])gpu', log_str)
        assert m is not None, f'Cannot parse gpu number in {log_str}'        
        gpu_number = int(m.group(1))
        return gpu_number

    if (unit == TestUnit.OVER_ALL_SINGLE_GPU 
        or unit == TestUnit.OVER_ALL_MULTI_GPU
    ):
        assert 'overall' in log_str, f"Log {log} is not overall"
        if unit == TestUnit.OVER_ALL_SINGLE_GPU:
            assert _parse_num_gpu(log_str) == 1, f"Log {log} is not single gpu"
        else:
            assert _parse_num_gpu(log_str) > 1, f"Log {log} is not multi gpu"
        return _parse_system_and_trace_helper(log)
    elif (unit == TestUnit.BREAKDOWN_SINGLE_GPU 
        or unit == TestUnit.BREAKDOWN_MULTI_GPU
    ):
        assert 'breakdown' in log_str, f"Log {log} is not breakdown"
        if unit == TestUnit.BREAKDOWN_SINGLE_GPU:
            assert _parse_num_gpu(log_str) == 1, f"Log {log} is not single gpu"
        else:
            assert _parse_num_gpu(log_str) > 1, f"Log {log} is not multi gpu"
        _, trace = _parse_system_and_trace_helper(log)
        if 'colsys' in log_str:
            system = 'colsys'
        elif 'strawman' in log_str:
            system = 'naive'
        return system, trace
    elif unit == TestUnit.ABLATION:
        assert False, f"should not use this function for ablation"
    elif unit == TestUnit.UNBALANCE:
        assert False, f"should not use this function for unbalance"
    elif unit == TestUnit.MEMORY_PRESSURE:
        assert False, f"should not use this function for memory pressure"
    elif unit == TestUnit.LLM:
        system, _ = _parse_system_and_trace_helper(log)
        return system, None
    else:
        raise ValueError(f"Unknown test unit: {unit}")
    

def _parse_train_local_thpts(log: pathlib.Path):
    train_thpt = log / 'train_thpt'
    if not train_thpt.exists():
        print(f'Warning: {log} does not have train_thpt')
        return np.nan, np.nan
    
    thpts = []
    with open(train_thpt, 'r') as f:
        for line in f.readlines():
            m = re.search(
                r'\[Rank (\d) \| [a-zA-Z-]+\] thpt: ([0-9]+\.[0-9]+) / ([0-9]+\.[0-9]+|nan) it/sec', 
                line)
            if m:
                rank = int(m.group(1))
                thpt = float(m.group(2))
                thpts.append((rank, thpt))
    thpts = sorted(thpts, key=lambda x: x[0])
    print('_parse_train_local_thpts [(rank, thpt)]', thpts)
    assert len(thpts) == thpts[-1][0] + 1, f"Log {log} does not have all ranks"
    return [thpt[1] for thpt in thpts]


def _parse_train_waste_compute(log: pathlib.Path):
    train_thpt = log / 'train_thpt'
    if not train_thpt.exists():
        print(f'Warning: {log} does not have train_thpt')
        return np.nan
    
    waste_compute = np.nan
    with open(train_thpt, 'r') as f:
        for line in f.readlines():
            m = re.search(
                r'cancel waste ([0-9]+\.[0-9]+) \(([0-9]+\.[0-9]+)%\).*\n^(?!.*global)', 
                line, re.MULTILINE)
            if m is not None:
                waste_compute = float(m.group(2))
                break
    return waste_compute


def _parse_infer_load_ltc(log: pathlib.Path):
    profile_log = log / 'profile-log.log'
    if not profile_log.exists():
        print(f'Warning: {log} does not have profile-log.log')
        return np.nan

    load = np.nan
    with open(profile_log, 'r') as f:
        for line in f.readlines():
            m = re.search(r'^InferLoadParam: avg ([0-9]+\.[0-9]+) ', line)
            if m:
                load = float(m.group(1))
                break
    return load


def _parse_infer_cold_start_ratio(log: pathlib.Path):
    profile_log = log / 'profile-log.log'
    if not profile_log.exists():
        print(f'Warning: {log} does not have profile-log.log')
        return np.nan
    
    num_cold_start = None
    num_infer = None
    with open(profile_log, 'r') as f:
        for line in f.readlines():
            m = re.search(r'^InferLoadParam: .* cnt ([0-9]+) ', line)
            if m is not None:
                num_cold_start = int(m.group(1))
            
            m = re.search(r'^InferJobProcess: .* cnt ([0-9]+) ', line)
            if m is not None:
                num_infer = int(m.group(1))
            
            if num_cold_start is not None and num_infer is not None:
                break
    if (num_cold_start is None) or (num_infer is None) or (num_infer == 0):
        return np.nan
    else:
        return num_cold_start / num_infer


def _parse_system_performance(system: str, trace: str, log: pathlib.Path):
    p99, slo, thpt = None, None, None
    # infer p99
    workload_log = log / 'workload-log'
    if workload_log.exists():
        with open(workload_log, 'r') as f:
            for line in f.readlines():
                if re.search(r'^p99 ', line):
                    _, p99, _ = line.split(maxsplit=2)
                    break

    # train thpt
    train_thpt_path = log / 'train_thpt'
    if train_thpt_path.exists():
        keyword = 'epoch e2e thpt: '
        with open(train_thpt_path, 'r') as f:
            for line in f.readlines():
                if keyword in line:
                    t = line.index(keyword)
                    thpt = line[t + len(keyword):].split(' ')[0]
                    break
    if system == 'Infer-Only' or system == 'infer-only':
        thpt = 0
    else:
        if thpt == "nan":
            print(log)
            thpts = _parse_train_local_thpts(log)
            thpt = f'{np.sum(thpts):.2f}' 

    # infer slo
    infer_slo = log / 'infer-slo.svg'
    if infer_slo.exists():
        with open(infer_slo, 'r') as f:
            content = f.read()
            m = re.search(
                r'^[ ]+<g +id="text_20">[ ]*\n[ ]*<!-- ([0-9]+\.[0-9]+) -->', 
                content, re.MULTILINE)

            if m:
                slo = m.group(1)
    else:
        print(f'Warning: {log} does not have infer-slo.svg')

    p99, slo, thpt = list(map(lambda x : x if x is not None else np.nan, 
                              [p99, slo, thpt]))
    return p99, slo, thpt


def _parse_llm_slo(log: pathlib.Path):
    infer_slo = log / 'infer-slo.svg'
    if not infer_slo.exists():
        print(f'Warning: {log} does not have infer-slo.svg')
        return np.nan

    ttft_slo = None
    tbt_slo = None
    with open(infer_slo, 'r') as f:
        content = f.read()
        m = re.search(
            r'^[ ]+<g +id="text_20">[ ]*\n[ ]*<!-- ([0-9]+\.[0-9]+) -->', 
            content, re.MULTILINE)
        if m:
            ttft_slo = m.group(1)

        m = re.search(
            r'^[ ]+<g +id="text_44">[ ]*\n[ ]*<!-- ([0-9]+\.[0-9]+) -->',
            content, re.MULTILINE)
        if m:
            tbt_slo = m.group(1)
    return ttft_slo, tbt_slo


# MARK: Overall Single GPU
def parse_over_all_single_gpu(logs: List[pathlib.Path]):
    if len(logs) == 0:
        print('Warning: over_all_single_gpu: logs are empty')
        return

    idx = []
    for sys in ['ColSys', 'Infer-Only', 'SP-I', 'SP-F', 'TaskSwitch', 'UM+MPS']:
        for metric in ['Infer', 'SLO', 'Train']:
            idx.append((sys, metric))

    # print (idx)
    df = pd.DataFrame(columns=['Light', 'Heavy', 'Burst', 'Skewed', 'MAF'],
                      index=pd.MultiIndex.from_tuples(idx))
    # print (df.sort_index(level=[1, 0]))

    for log in logs:
        system, trace = _parse_system_and_trace(TestUnit.OVER_ALL_SINGLE_GPU, log)
        if system is None or trace is None:
            print(f'Warning: {log} does not match any system or trace')
            continue
        p99, slo, thpt = _parse_system_performance(system, trace, log)

        print(f'Parsed {log}: {system}, {trace}, {p99}, {slo}, {thpt}')
        df.loc[(system, 'Infer'), trace] = p99
        df.loc[(system, 'Train'), trace] = thpt
        df.loc[(system, 'SLO'), trace] = slo

    log_parent_dir = logs[0].parent
    print(log_parent_dir)
    with open(log_parent_dir / 'overall_v100.dat', 'w') as f:
        sorted_df = df.sort_index(level=[1, 0])
        print(sorted_df, file=f)
    
    try:
        print(f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/overall_v100.py at \033[4:5m{log_parent_dir}\033[24m\033[0m')
        subprocess.run(['python', f'{get_plot_script_path("overall_v100")}'],
                    cwd=log_parent_dir)
    except Exception as e:
        print(f"Error executing plot script: {e}")
        raise
    

# MARK: Overall Multi GPU
def parse_over_all_multi_gpu(logs: List[pathlib.Path]):
    if len(logs) == 0:
        print('Warning: over_all_multi_gpu: logs are empty')
        return

    idx = ['SP-50', 'SP-75', 'ColSys', 'TaskSwitch', 'UM+MPS', 'Infer-Only']
    columns = ['Infer', 'Train', 'SLO']
    df = pd.DataFrame(columns=columns, index=idx)

    for log in logs:
        system, trace = _parse_system_and_trace(TestUnit.OVER_ALL_MULTI_GPU, log)
        if system is None or trace is None:
            print(f'Warning: {log} does not match any system or trace')
            continue
        system = static_partition_to_number_name(system)

        assert trace == 'Light', f"Log {log} is not Light trace"
        p99, slo, thpt = _parse_system_performance(system, trace, log)
        df.loc[system, 'Infer'] = p99
        df.loc[system, 'Train'] = thpt
        df.loc[system, 'SLO'] = slo

    log_parent_dir = logs[0].parent
    with open(log_parent_dir / 'multi_gpu.dat', 'w') as f:
        print(df, file=f)
    
    try:
        print(f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/multi_gpu.py at \033[4:5m{log_parent_dir}\033[24m\033[0m')
        subprocess.run(['python', f'{get_plot_script_path("multi_gpu")}'],
                        cwd=log_parent_dir)
    except Exception as e:
        print(f"Error executing plot script: {e}")
        raise


# MARK: Breakdown
def parse_breakdown(logs: List[pathlib.Path], single_gpu: bool = True):
    if len(logs) == 0:
        print('Warning: parse_breakdown: logs are empty')
        return

    for log in logs:
        system, trace = _parse_system_and_trace(
            TestUnit.BREAKDOWN_SINGLE_GPU if single_gpu else TestUnit.BREAKDOWN_MULTI_GPU, 
            log
        )
        if system is None or trace is None:
            print(f'Warning: {log} does not match any system or trace')
            continue
        assert trace == 'MAF' or trace == 'azure', f"Log {log} is not MAF trace"
        
        profile_log = log / 'profile-log.log'
        if not profile_log.exists():
            print(f'Warning: {log} does not have profile-log.log')
            continue
    
        adjust_info = None
        with open(profile_log, 'r') as f:
            adjust_info = re.search(r'^\[Adjust Info\]\n((?s:.*))', f.read(), 
                                   re.MULTILINE)
            if adjust_info is None:
                print(f'Warning: {profile_log} does not contain adjust info')
            else:
                adjust_info = adjust_info.group(1)
                
        log_parent_dir = log.parent
        if adjust_info is not None:
            with open(log_parent_dir / f'adjust_breakdown_{system}_{"single" if single_gpu else "multi"}_gpu.dat', 'w') as f:
                f.write(adjust_info)

    try:
        print(f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/adjust_breakdown_single_or_multi_gpu.py --single-gpu at \033[4:5m{log_parent_dir}\033[24m\033[0m' 
              if single_gpu 
              else f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/adjust_breakdown_single_or_multi_gpu.py --multi-gpu at \033[4:5m{log_parent_dir}\033[24m\033[0m')
        subprocess.run(['python', f'{get_plot_script_path("adjust_breakdown_single_or_multi_gpu")}', 
                        '--single-gpu' if single_gpu else '--multi-gpu'],
                        cwd=log_parent_dir)
    except Exception as e:
        print(f"Error executing plot script: {e}")
        raise


def _parse_ablation_watermark(logs: List[pathlib.Path], parent_dir: pathlib.Path):
    df = pd.DataFrame(columns=['CacheSize', 'P99', 'Load', 'Thpt', 'WasteCompute'])

    if len(logs) == 0:
        print('Warning: parse_ablation_watermark: logs are empty')
        df.loc[0] = [0, np.nan, np.nan, np.nan, np.nan]
    else:
        df = df.set_index('CacheSize')
        for log in logs:
            m = re.search(r'((\d+(\.\d+)?)GB)-((\d+(\.\d+)?)GB)', log.name)
            if m is None:
                print(f'Warning: cannot parse cache size from {log}')
                continue
            cache_size = m.group(5)
            p99, slo, thpt = _parse_system_performance('ColSys', None, log)
            df.loc[cache_size, 'P99'] = p99
            df.loc[cache_size, 'Thpt'] = thpt
            df.loc[cache_size, 'Load'] = f'{_parse_infer_load_ltc(log):.1f}'
            df.loc[cache_size, 'WasteCompute'] = f'{_parse_train_waste_compute(log):.2f}'

        df = df.sort_index()
        df = df.reset_index()

    with open(parent_dir / 'ablation_cache_size.dat', 'w') as f:
        print(df.to_string(index=False), file=f)


def _parse_ablation_idle_time(logs: List[pathlib.Path], parent_dir: pathlib.Path):
    df = pd.DataFrame(columns=['MaxIdleMs', 'ColdStart', 'P99', 'Thpt', 'WasteCompute'])

    if len(logs) == 0:
        print('Warning: parse_ablation_idle_time: logs are empty')
        df.loc[0] = [0, np.nan, np.nan, np.nan, np.nan]
    else:
        df = df.set_index('MaxIdleMs')
        for log in logs:
            m = re.search(r'(\d+)ms', log.name)
            if m is None:
                print(f'Warning: cannot parse idle time from {log}')
                continue
            idle_time = float(m.group(1))
            p99, slo, thpt = _parse_system_performance('ColSys', None, log)
            df.loc[idle_time, 'P99'] = p99
            df.loc[idle_time, 'Thpt'] = thpt
            df.loc[idle_time, 'ColdStart'] = f'{_parse_infer_cold_start_ratio(log):.2f}'
            df.loc[idle_time, 'WasteCompute'] = f'{_parse_train_waste_compute(log):.2f}'

        df = df.sort_index()
        df = df.reset_index()
        df['MaxIdleMs'] = df['MaxIdleMs'].apply(lambda x: f'{x / 1000:.1f}')

    with open(parent_dir / 'ablation_infer_idle_time.dat', 'w') as f:
        print(df.to_string(index=False), file=f)
    

# MARK: Ablation
def parse_ablation(logs: List[pathlib.Path]):
    if len(logs) == 0:
        print('Warning: parse_ablation: logs are empty')
        return

    watermark_ablation_logs = []
    idle_time_ablation_logs = []

    for log in logs:
        if 'ablation-cold-cache' in str(log):
            watermark_ablation_logs.append(log)
        elif 'ablation-infer-idle-time' in str(log):
            idle_time_ablation_logs.append(log)
    
    _parse_ablation_watermark(watermark_ablation_logs, logs[0].parent)
    _parse_ablation_idle_time(idle_time_ablation_logs, logs[0].parent)

    try:
        print(f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/ablation.py at \033[4:5m{logs[0].parent}\033[24m\033[0m')
        subprocess.run(['python', f'{get_plot_script_path("ablation")}'],
                        cwd=logs[0].parent)
    except Exception as e:
        print(f"Error executing plot script: {e}")
        raise


# MARK: Unbalance
def parse_unbalance(logs: List[pathlib.Path]):
    if len(logs) == 0:
        print('Warning: parse_unbalance: logs are empty')
        return

    df = pd.DataFrame(columns=['GPU0', 'GPU1'], index=['BAL', 'IMBAL'])
    for log in logs:
        if 'colsys-balance' in log.name:
            idx = 'BAL'
        elif 'colsys-imbalance' in log.name:
            idx = 'IMBAL'
        else:
            print(f'Warning: {log} does not match any')
            continue

        thpt1, thpt2 = _parse_train_local_thpts(log)
        df.loc[idx, 'GPU0'] = thpt1
        df.loc[idx, 'GPU1'] = thpt2

        log_parent_dir = log.parent    
        profile_log = log / 'profile-log.log'
        assert profile_log.exists(), f"Log {log} does not have profile-log.log"
        if idx == 'BAL':
            subprocess.run(["cp", profile_log, f"{log_parent_dir / 'memory_util_gpus_bal.dat'}"])
        else:
            subprocess.run(["cp", profile_log, f"{log_parent_dir / 'memory_util_gpus_imbal.dat'}"])

    log_parent_dir = logs[0].parent
    with open(log_parent_dir / 'memory_util_gpus_thpt.dat', 'w') as f:
        print(df, file=f)

    try:
        print(f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/memory_util_gpus.py at \033[4:5m{log_parent_dir}\033[24m\033[0m')
        subprocess.run(['python', f'{get_plot_script_path("memory_util_gpus")}'],
                        cwd=log_parent_dir)
    except Exception as e:
        print(f"Error executing plot script: {e}")
        raise


# MARK: Memory Pressure
def parse_memory_pressure(logs):
    if len(logs) == 0:
        print('Warning: parse_memory_pressure: logs are empty')
        return
    assert len(logs) == 1, f"Memory pressure only use one log"
    assert 'memory-pressure' in str(logs[0]), f"Log {logs[0]} is not memory-pressure"

    workload_log = logs[0] / 'workload-log'
    assert workload_log.exists(), f"Log {logs[0]} does not have workload-log"

    train_profile = logs[0] / 'train-profile.csv'
    assert train_profile.exists(), f"Log {logs[0]} does not have train-profile.csv"

    log_parent_dir = logs[0].parent
    try:
        subprocess.run(['python', 'util/profile/collect_infer_ltc.py', 
                    '-l', str(workload_log), 
                    '-o', str(log_parent_dir / "memory_pressure_ltc.dat")])
        subprocess.run(['python', 'util/profile/collect_train_thpt.py',
                       '-l', str(train_profile),
                       '-o', str(log_parent_dir / "memory_pressure_thpt.dat")])
        
        print(f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/memory_pressure.py at \033[4:5m{log_parent_dir}\033[24m\033[0m')
        subprocess.run(['python', f'{get_plot_script_path("memory_pressure")}'],
                        cwd=log_parent_dir)
    except Exception as e:
        print(f"Error executing plot script: {e}")
        raise


# MARK: LLM
def parse_llm(logs: List[pathlib.Path]):
    if len(logs) == 0:
        print('Warning: parse_llm: logs are empty')
        return
    
    df = pd.DataFrame(columns=['TTFT-SLO', 'TBT-SLO', 'Train'],
                      index=['SP-50', 'SP-75', 'ColSys', 'Infer-Only'])
    for log in logs:
        system, _ = _parse_system_and_trace(TestUnit.LLM)
        if system is None:
            print(f'Warning: {log} does not match any system')
            continue
        system = static_partition_to_number_name(system)

        _, _, thpt = _parse_system_performance(system, None, log)

        ttft_slo, tbt_slo = _parse_llm_slo(log)
        df.loc[system, 'TTFT-SLO'] = ttft_slo
        df.loc[system, 'TBT-SLO'] = tbt_slo
        df.loc[system, 'Train'] = thpt

    with open(logs[0].parent / 'llm.dat', 'w') as f:
        print(df, file=f)

    try:
        print(f'\033[1m\033[34mExecuting plot script: eval/runner/plot_scripts/llm.py at \033[4:5m{logs[0].parent}\033[24m\033[0m')
        subprocess.run(['python', f'{get_plot_script_path("llm")}'],
                        cwd=logs[0].parent)
    except Exception as e:
        print(f"Error executing plot script: {e}")
        raise


class LogParser:
    _logs = []
    _enable = False
    
    @classmethod
    def clear_logs(cls):
        cls._logs.clear()

    @classmethod
    def add_log(cls, log):
        cls._logs.append(log)

    @classmethod
    def parse(cls, unit:TestUnit, verbose:bool=True):
        if verbose:
            print(f"Parsing logs for unit: {unit}, Logs:")
            print(f'    >>> python eval/runner/parser/log_parser.py '
                  f'--unit {unit.name} --logs {" ".join(str(log) for log in cls._logs)}')
            print('')

        if unit == TestUnit.OVER_ALL_SINGLE_GPU:
            return parse_over_all_single_gpu(cls._logs)
        elif unit == TestUnit.OVER_ALL_MULTI_GPU:
            return parse_over_all_multi_gpu(cls._logs)
        elif unit == TestUnit.BREAKDOWN_SINGLE_GPU:
            return parse_breakdown(cls._logs, True)
        elif unit == TestUnit.BREAKDOWN_MULTI_GPU:
            return parse_breakdown(cls._logs, False)
        elif unit == TestUnit.ABLATION:
            return parse_ablation(cls._logs)
        elif unit == TestUnit.UNBALANCE:
            return parse_unbalance(cls._logs)
        elif unit == TestUnit.MEMORY_PRESSURE:
            return parse_memory_pressure(cls._logs)
        elif unit == TestUnit.LLM:
            return parse_llm(cls._logs)
        else:
            raise ValueError(f"Unknown test unit: {unit}")
        

# ==========================================================
# ==========================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Parse log files')
    parser.add_argument('--logs', type=str, nargs='+',
                        help='List of log files to parse', required=True)
    parser.add_argument('--unit', type=str, choices=[e.name for e in TestUnit], 
                        help='Test unit to parse', required=True)
    args = parser.parse_args()

    logs = [pathlib.Path(log) for log in args.logs]
    for log in logs:
        LogParser.add_log(log)
    unit = TestUnit[args.unit]

    LogParser.parse(unit)