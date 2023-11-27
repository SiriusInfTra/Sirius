from runner import System, HyperWorkload, _global_seed
from runner import *

system = System(mode=System.ServerMode.Normal, use_sta=False, mps=True)
workload = HyperWorkload(concurrency=512, seed=_global_seed, delay_before_infer=0)
workload.infer_workloads.append(PoissonInferWorkload(
    poisson_params=zip([InferModel("resnet152")], [PoissonParam(0, 64)]),
    duration=30, 
    seed=_global_seed
))
# workload.train_workload = TrainWorkload('resnet', 15, 60)

system.launch("um-mps-issue", "1", time_stamp=False)
workload.launch(system)
system.stop()