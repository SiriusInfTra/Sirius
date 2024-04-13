import os
import argparse
from runner import *
from dataclasses import dataclass
import run_comm

run_comm.use_time_stamp = True
run_comm.retry_if_fail = False
run_comm.skip_fail = False

# run_comm.fake_launch = True

set_global_seed(42)


