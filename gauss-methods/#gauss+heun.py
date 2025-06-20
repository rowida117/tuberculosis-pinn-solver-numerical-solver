
#gauss+heun

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time

# Parameters
params = {
    'Lambda': 2.0,
    'beta': 0.025,
    'delta': 1.0,
    'p': 0.3,
    'mu': 0.0101,
    'k': 0.005,
    'r1': 0.0,
    'r2': 0.8182,
    'phi': 0.02,
    'gamma': 0.01,
    'd1': 0.022722,
    'd2': 0.20
}

# TB ODE system
def TB_ODE_system(t, y):
    S, E, I, L = y
    Lambda = params['Lambda']
    beta = params['beta']
    delta = params['delta']
    p = params['p']
    mu = params['mu']
    k = params['k']
    r1 = params['r1']
    r2 = params['r2']
    phi = params['phi']
    gamma = params['gamma']
    d1 = params['d1']
    d2 = params['d2']

    dS = Lambda - beta * S * (I + delta * L) - mu * S
    dE = beta * (1 - p) * S * (I + delta * L) + r2 * I - (mu + k * (1 - r1)) * E
    dI = beta * p * S * (I + delta * L) + k * (1 - r1) * E + gamma * L - (mu + d1 + phi * (1 - r2) + r2) * I
    dL = phi * (1 - r2) * I - (mu + d2 + gamma) * L
    return np.array([dS, dE, dI, dL])

def gauss_integrate_vector_2pt(f, a, b, y_n, h_heun=0.1):
    # 2-point Gauss quadrature nodes and weights
    x = np.array([-0.5773502691896257, 0.5773502691896257])  # nodes
    w = np.array([1.0, 1.0])  # weights
    
    numSteps = 2
    mid = (a + b) / 2
    half_width = (b - a) / 2

    result = np.zeros_like(y_n)
    for i in range(2):
        t_i = mid + half_width * x[i]
        result += w[i] * f(t_i, y_n)
        numSteps += 3  # +1 for the f(t_i, y_ti) evaluation
    
    return half_width * result, numSteps


# Time and initial conditions
t_start = 0
t_end = 20
dt = 0.5
times = np.arange(t_start, t_end + dt, dt)
y0 = np.array([params['Lambda'] / params['mu'], 1.0, 0.0, 0.0])
start_time = time.time()

# Main integration loop
results = [y0]
totalFinalCount = 0
for i in range(1, len(times)):
    t_prev = times[i - 1]
    t_curr = times[i]
    y_prev = results[-1]
    integral, steps = gauss_integrate_vector_2pt(TB_ODE_system, t_prev, t_curr, y_prev)
    totalFinalCount += steps
    y_next = y_prev + integral
    results.append(y_next)
end_time = time.time()


print("Number of steps total:", totalFinalCount)
# DataFrame
results_array = np.array(results)
df_full = pd.DataFrame({
    'Time': times,
    'S': results_array[:, 0],
    'E': results_array[:, 1],
    'I': results_array[:, 2],
    'L': results_array[:, 3]
})

# Output as table
print("\nFull TB System Solved with RK4-enhanced 6-Point Gauss Quadrature")
print(df_full.to_string(index=False))
print("Execution time: {:.6f} seconds".format(end_time - start_time))

# Plotting 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=True)

axs[0, 0].plot(df_full['Time'], df_full['S'], color='tab:blue', linewidth=2)
axs[0, 0].set_ylabel('S (Susceptible)')
axs[0, 0].set_title('TB Model Compartments (RK4 + Gauss Quadrature)')
axs[0, 0].grid(True)

axs[0, 1].plot(df_full['Time'], df_full['E'], color='tab:orange', linewidth=2)
axs[0, 1].set_ylabel('E (Exposed)')
axs[0, 1].grid(True)

axs[1, 0].plot(df_full['Time'], df_full['I'], color='tab:red', linewidth=2)
axs[1, 0].set_ylabel('I (Infectious)')
axs[1, 0].set_xlabel('Time (years)')
axs[1, 0].grid(True)

axs[1, 1].plot(df_full['Time'], df_full['L'], color='tab:green', linewidth=2)
axs[1, 1].set_ylabel('L (Latent)')
axs[1, 1].set_xlabel('Time (years)')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
