import time
import argparse
import pycolserve
import random

parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=int, default=1000) # ms
args = parser.parse_args()
interval = args.interval

cmd_mq = pycolserve.PyMemoryQueue('cmd-ctrl', True)
status_mq = pycolserve.PyMemoryQueue('status-ctrl', True)

input('Press Enter to start...')

while True:
	i = random.random() * interval
	time.sleep(i / 1000)
	cmd_mq.put(pycolserve.Event.kColocateAdjustL1)
	t0 = time.time()
	event = None
	while event is None:
		event = status_mq.timed_get(10)
	t1 = time.time()
	assert event == pycolserve.Event.kColocateAdjustL1Done
	print('colocate adj l1 {:.1f}ms'.format((t1 - t0) * 1000))
        
	
	
