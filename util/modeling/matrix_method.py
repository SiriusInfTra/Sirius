import scipy
import numpy as np
import math


class hypoexp:
    def __init__(self, lambdas):
        self.lambdas = lambdas
        self._Theta = np.zeros((len(lambdas), len(lambdas)))
        for i in range(len(lambdas)):
            self._Theta[i, i] = -lambdas[i]
            if i + 1 != len(lambdas):
                self._Theta[i, i + 1] = lambdas[i]
        
    def cdf(self, x):
        tail_prob = scipy.linalg.expm(x * self._Theta)[0, :] @ np.ones(len(self.lambdas))
        return 1 - tail_prob


rps = 5 / 56                      # \lambda
service_time = 0.0084                # E[S]
service_rate = 1 / service_time     # \mu
live_time = 5                       # t
cold_start_time = 0.0159             # L
cold_start_rate = 1 / cold_start_time
hite_rate = 1 / 12                  # a

slo = 0.0293                         # SLO

L_0 = np.array([
    [-(1 / live_time + rps),  1 / live_time],
    [0,                       -rps         ]
])
print(f'Matrix L_0: \n', L_0, end='\n\n')

L = np.array([
    [-(rps + service_rate), 0                           ],
    [1 / cold_start_time,   -(rps + 1 / cold_start_time)]
])

B = np.array([
    [rps, 0],
    [0,   0]
])
print(f'Matrix B: \n', B, end='\n\n')

F_0 = np.array([
    [rps,             0                   ],
    [hite_rate * rps, (1 - hite_rate) * rps]
])
print(f'Matrix F_0: \n', F_0, end='\n\n')

F = np.array([
    [rps,   0],
    [0,   rps]
])

L_rev = np.linalg.inv(L)

# begin iteration
R_0 = np.zeros((2, 2))
R_1 = - (R_0 @ R_0 @ B + F) @ L_rev
# while np.linalg.norm(R_1 - R_0, ord=) > 1e-8:
while np.max(np.abs(R_1 - R_0)) > 1e-8:
    R_0 = R_1
    R_1 = - (R_0 @ R_0 @ B + F) @ L_rev

print(f'Matrix R: \n', R_1, end='\n\n')
R = R_1


Phi_11 = L + R @ B
Phi = np.block([
    [L_0, F_0],
    [B,   Phi_11]
])

Psi = np.block([
    [np.ones((2, 1))],
    [np.linalg.inv((np.eye(2) - R)) @ np.ones((2, 1))]
])

# Psi = np.linalg.inv(np.eye(2) - R) @ np.ones((2, 1))
print(f'Matrix Phi {Phi.shape}: \n', Phi, end='\n\n')
print(f'Matrix Psi {Psi.shape}: \n', Psi, end='\n\n')

# E = np.array([
#     [Psi[0, 0], Phi[0, 1]],
#     [Psi[1, 0], Phi[1, 1]]
# ])
E = np.hstack([Psi, Phi[:, 1:]])
print(f'Matrix E: \n', E, end='\n\n')

# pi0 = np.array([1, 0]) @ np.linalg.inv(E)
# pi1 = 
init_pi = np.array([1, 0, 0, 0]) @ np.linalg.inv(E)
pi_0, pi_1 = init_pi[:2], init_pi[2:]

# pi_1[1] = 0

# # pi = [i_warm, i_cold]
print(f'init_pi = {init_pi}, \n\npi_0 = {pi_0}, \npi_1 = {pi_1}')

pi_list = [pi_0, pi_1]

R_power_i = R
upper_bound = 20
for i in range(2, upper_bound):
    pi_i = pi_1 @ R_power_i
    R_power_i = R_power_i @ R
    print(f'pi_{i} = {pi_i}')
    pi_list.append(pi_i)


prob = 0
for i in range(upper_bound):
    pi = pi_list[i]
    pi_w, pi_c = pi
    if i == 0:
        # warm
        prob += pi_w * (1 - math.exp(-service_rate * slo))
        # cold
        prob += pi_c * (
            hite_rate * (1 - math.exp(-service_rate * slo)) +
            (1 - hite_rate) * hypoexp([service_rate, cold_start_rate]).cdf(slo)
        )
    else:
        # warm
        prob += pi_w * hypoexp([service_rate] * (i + 1)).cdf(slo)
        # cold
        prob += pi_c * hypoexp([service_rate] * (i + 1) + [cold_start_rate]).cdf(slo)

print(f'P(slo) = {prob}')




