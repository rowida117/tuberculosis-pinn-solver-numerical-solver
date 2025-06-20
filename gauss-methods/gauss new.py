import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import time

#2 point

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

# Define the full coupled TB ODE system using parameters and Gauss Quadrature

# Define the TB model ODEs
def TB_ODE_system(t, y):
      # Parameters from the TB model
      # Extract variables from y
    S, E, I, L = y
   
    # Extract parameters
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

# Use 2-point Gauss Quadrature to integrate vector ODE over [a, b]
def gauss_integrate_vector(f, a, b, y):
    x0 = -1 / math.sqrt(3)
    x1 = 1 / math.sqrt(3)
    w0 = w1 = 1
   
    f0 = f(t0, y)
    f1 = f(t1, y)
    return (b - a) / 2 * (w0 * f0 + w1 * f1)

# Time setup and initial condition (Case 2: endemic, small perturbation)
t_start = 0
t_end = 20
dt = 0.01
times = np.arange(t_start, t_end + dt, dt)
y0 = np.array([params['Lambda'] / params['mu'], 1.0, 0.0, 0.0])

# Store results
results = [y0]

start_time=time.time()
# Integrate using Gauss Quadrature
for i in range(1, len(times)):
    t_prev = times[i - 1]
    t_curr = times[i]
    y_prev = results[-1]
    integral = gauss_integrate_vector(TB_ODE_system, t_prev, t_curr, y_prev)
    y_next = y_prev + integral
    results.append(y_next)
end_time=time.time()

# Format and display results
results_array = np.array(results)
df_full = pd.DataFrame({
    'Time': times,
    'S': results_array[:, 0],
    'E': results_array[:, 1],
    'I': results_array[:, 2],
    'L': results_array[:, 3]
})

print("Full TB System Solved with 2-Point Gauss Quadrature", df_full)
print("Execution time: {:.6f} seconds".format(end_time - start_time))




# Step 1: Load reference solution from Excel
ref_df = pd.read_excel(r"C:\Users\sheri\Downloads\TB_Model_Output.xlsx")
ref_df.columns = ['Time', 'S', 'E', 'I', 'L']

# Step 2: Merge reference data with Gauss result data
df_plot = pd.merge(df_full, ref_df, on='Time', how='inner')

# Step 4: Calculate Mean Squared Error for each compartment
mse_S = np.mean((df_plot['S_x'] - df_plot['S_y']) ** 2)
mse_E = np.mean((df_plot['E_x'] - df_plot['E_y']) ** 2)
mse_I = np.mean((df_plot['I_x'] - df_plot['I_y']) ** 2)
mse_L = np.mean((df_plot['L_x'] - df_plot['L_y']) ** 2)

print("\nMean Squared Errors (MSE) compared to Reference:")
print(f"S (Susceptible): {mse_S:.6f}")
print(f"E (Exposed)    : {mse_E:.6f}")
print(f"I (Infectious) : {mse_I:.6f}")
print(f"L (Latent)     : {mse_L:.6f}")

#df_plot.to_excel(r"C:\Users\sheri\Documents\TB_Model_Comparison.xlsx", index=False)

# Step 3: Create subplots and plot each variable with reference
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)


# S compartment
axs[0, 0].plot(df_full['Time'], df_full['S'], color='tab:blue', linewidth=2, label='S - Gauss 2-Point')
axs[0, 0].plot(ref_df['Time'], ref_df['S'], color='black', linestyle='--', linewidth=1, label='S - Reference')
axs[0, 0].set_ylabel('S (Susceptible)')
axs[0, 0].set_title('TB Model Compartments')
axs[0, 0].grid(True)
axs[0, 0].legend()

# E compartment
axs[0, 1].plot(df_full['Time'], df_full['E'], color='tab:orange', linewidth=2, label='E - Gauss 2-Point')
axs[0, 1].plot(ref_df['Time'], ref_df['E'], color='black', linestyle='--', linewidth=1, label='E - Reference')
axs[0, 1].set_ylabel('E (Exposed)')
axs[0, 1].grid(True)
axs[0, 1].legend()

# I compartment
axs[1, 0].plot(df_full['Time'], df_full['I'], color='tab:red', linewidth=2, label='I - Gauss 2-Point')
axs[1, 0].plot(ref_df['Time'], ref_df['I'], color='black', linestyle='--', linewidth=1, label='I - Reference')
axs[1, 0].set_ylabel('I (Infectious)')
axs[1, 0].set_xlabel('Time (years)')
axs[1, 0].grid(True)
axs[1, 0].legend()

# L compartment
axs[1, 1].plot(df_full['Time'], df_full['L'], color='tab:green', linewidth=2, label='L - Gauss 2-Point')
axs[1, 1].plot(ref_df['Time'], ref_df['L'], color='black', linestyle='--', linewidth=1, label='L - Reference')
axs[1, 1].set_ylabel('L (Latent)')
axs[1, 1].set_xlabel('Time (years)')
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout()
plt.show()
