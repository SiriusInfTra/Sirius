import scipy
import numpy as np
import math

rps = 100 / 56                      # \lambda
service_time = 0.010                # E[S]
service_rate = 1 / service_time     # \mu
live_time = 0.001                       # t
cold_start_time = 0.050             # L
hite_rate = 0 / 12                  # a

slo = 0.030                         # SLO

L_0 = np.array([
    [-(1 / live_time + rps),  1 / live_time],
    [0,                       -rps         ]
])

L = np.array([
    [-(rps + service_rate), 0                           ],
    [1 / cold_start_time,   -(rps + 1 / cold_start_time)]
])

B = np.array([
    [rps, 0],
    [0,   0]
])

F_0 = np.array([
    [rps,             0                   ],
    [hite_rate * rps, (1 - hite_rate) * rps]
])

F = np.array([
    [rps,   0],
    [0,   rps]
])

L_rev = np.linalg.inv(L)

# begin iteration
R_0 = np.zeros((2, 2))
R_1 = - (R_0 @ R_0 @ B + F) @ L_rev
while np.linalg.norm(R_1 - R_0) > 1e-7:
    R_0 = R_1
    R_1 = - (R_0 @ R_0 @ B + F) @ L_rev

print(f'Matrix R: \n', R_1, end='\n\n')
R = R_1

Phi = L_0 + R_1 @ B
Psi = np.linalg.inv(np.eye(2) - R) @ np.ones((2, 1))
print(f'Matrix Phi: \n', Phi, end='\n\n')
print(f'Matrix Psi: \n', Psi, Psi.shape, end='\n\n')

E = np.array([
    [Psi[0, 0], Phi[0, 1]],
    [Psi[1, 0], Phi[1, 1]]
])

pi0 = np.array([1, 0]) @ np.linalg.inv(E)

# pi = [i_warm, i_cold]
print(f'pi_0 = {pi0}')

pi_list = [pi0]

R_power_i = R
upper_bound = 10
for i in range(1, upper_bound):
    pi_i = pi0 @ R_power_i
    R_power_i = R_power_i @ R
    print(f'pi_{i} = {pi_i}')
    pi_list.append(pi_i)


prob = 0
for i in range(upper_bound):
    pi = pi_list[i]
    pi_w, pi_c = pi
    if i == 0:
        # warm
        if service_time < slo:
            prob += pi_w
        # cold
        if service_time + cold_start_time < slo:
            prob += pi_c
    else:
        # warm
        if service_time * (i - 1) + 




