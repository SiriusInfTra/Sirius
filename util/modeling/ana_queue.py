import scipy
import numpy as np
import math

rps = 100


def warm_cold_num_busy_period_ratio(rps, hit_rate, live_time):
    x = (1 - hit_rate) * math.exp(-rps * live_time)
    warm = 1 - x
    cold = x
    print(f'hit_rate {hit_rate}, #warm_period : #cold_period: {warm} : {cold}')
    return warm, cold


def avg_num_req_per_warm_period(rps, service_time):
    assert rps * service_time < 1
    return 1 / (1 - rps * service_time)


def avg_num_req_per_cold_period(rps, service_time, cold_start_time):
    return (1 + rps * cold_start_time) / (1 - rps * service_time)


def warm_cold_num_req_ratio(rps, service_time, hit_rate, live_time, cold_start_time):
    warm, cold = warm_cold_num_busy_period_ratio(rps, hit_rate, live_time)
    warm_req = avg_num_req_per_warm_period(rps, service_time) * warm
    cold_req = avg_num_req_per_cold_period(rps, service_time, cold_start_time) * cold
    return warm_req, cold_req
    

def avg_warm_period_queue_time(rps, service_time):
    # assume service time is exponential
    E_S2 = 2 * service_time ** 2
    ret = rps * E_S2 / (2 * (1 - rps * service_time))
    print(f'avg_warm_period_queue_time {ret}')
    return ret


def avg_warm_period_in_system_time(rps, service_time):
    return avg_warm_period_queue_time(rps, service_time) + service_time


def avg_cold_period_queue_time(rps, service_time, cold_start_time):
    # assume service time is exponential
    E_S2 = 2 * service_time ** 2
    # assume cold start time is exponential
    E_I2 = 2 * cold_start_time ** 2
    rho = rps * service_time
    assert rho < 1

    # due to the assumption of exponential cold start time,
    # this should equal warm period queue time + cold period queue time
    ret = (
        (rps * E_S2) / (2 * (1 - rho)) 
        + (2 * cold_start_time + rps * E_I2) /  (2 * (1 + rps * cold_start_time))
    )
    print(f'avg_cold_period_queue_time {ret}')
    return ret


def avg_cold_period_in_system_time(rps, service_time, cold_start_time):
    # a0 prob
    a0 = 1 / avg_num_req_per_cold_period(rps, service_time, cold_start_time)
    return (
        avg_cold_period_queue_time(rps, service_time, cold_start_time) + 
        service_time + 
        a0 * cold_start_time
    )


def avg_queue_time(rps, service_time, hit_rate, live_time, cold_start_time):
    warm_req, cold_req = warm_cold_num_req_ratio(
        rps, service_time, hit_rate, live_time, cold_start_time)
    warm_req_ratio = warm_req / (warm_req + cold_req)
    cold_req_ratio = cold_req / (warm_req + cold_req)
    avg_queue_time = (
        warm_req_ratio * avg_warm_period_in_system_time(rps, service_time) +
        cold_req_ratio * avg_cold_period_in_system_time(rps, service_time, cold_start_time)
    )
    return avg_queue_time


rps = 5 / 56
service_time = 0.010
hit_rate = 2 / 12
live_time = 5
cold_start_time = 0.050

print(avg_queue_time(rps, service_time, hit_rate, live_time, cold_start_time))