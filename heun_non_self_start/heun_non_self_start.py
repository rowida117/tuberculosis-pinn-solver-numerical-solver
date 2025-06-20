
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# === Load true values from Excel ===
real_data = generate_reference_solution()
t_real = real_data["Time"]
S_real = real_data["S"]
E_real = real_data["E"]
I_real = real_data["I"]
L_real = real_data["L"]

# === Parameters (from R code) ===
params = {
    'lambda_': 2,
    'beta': 0.025,
    'delta': 1,
    'p': 0.3,
    'mu': 0.0101,
    'k': 0.005,
    'r1': 0,
    'r2': 0.8182,
    'phi': 0.02,
    'gamma': 0.01,
    'd1': 0.0227,
    'd2': 0.20
}

# === ODE function ===
ode_call_count = 0

def tb_model(t, y, p):
    global ode_call_count
    ode_call_count += 1

    S, E, I, L = y
    λ = p['lambda_']
    β = p['beta']
    δ = p['delta']
    μ = p['mu']
    k = p['k']
    r1 = p['r1']
    r2 = p['r2']
    φ = p['phi']
    γ = p['gamma']
    d1 = p['d1']
    d2 = p['d2']
    p_val = p['p']

    dS = λ - β * S * (I + δ * L) - μ * S
    dE = β * (1 - p_val) * S * (I + δ * L) + r2 * I - (μ + k * (1 - r1)) * E
    dI = β * p_val * S * (I + δ * L) + k * (1 - r1) * E + γ * L - (μ + d1 + φ * (1 - r2) + r2) * I
    dL = φ * (1 - r2) * I - (μ + d2 + γ) * L

    return np.array([dS, dE, dI, dL])

# === Heun Non-Self-Starting Method ===
def heun_non_self_start(f, y0, t, p):
    N = len(t)
    y = np.zeros((N, len(y0)))
    y[0] = y0
    y[1] = y[0] + (t[1] - t[0]) * f(t[0], y[0], p)

    for n in range(1, N - 1):
        h = t[n+1] - t[n]
        fn = f(t[n], y[n], p)
        y_predict = y[n] + h * fn
        f_predict = f(t[n+1], y_predict, p)
        y[n+1] = y[n] + (h / 2) * (fn + f_predict)

    return y

# === Initial Conditions ===
y0 = [params['lambda_'] / params['mu'], 1, 0, 0]  # epidemic case
t = np.linspace(0, 20, 41)

# === Solve the system ===
start_time = time.time()
solution_heun = heun_non_self_start(tb_model, y0, t, params)
end_time = time.time()

execution_time = (end_time - start_time) * 1000  # in ms
S_heun, E_heun, I_heun, L_heun = solution_heun.T

# === Print results ===
print(f"\n📈 Total ODE function calls: {ode_call_count}")
print(f"⏱️ Execution Time: {execution_time:.6f} ms\n")

# === Plotting with true values ===
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(t, S_heun, 'b', linewidth=2, label='Heun S(t)')
plt.plot(t_real, S_real, 'b--', label='True S(t)')
plt.title("Susceptible"); plt.xlabel("Time"); plt.ylabel("S")
plt.grid(True); plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, E_heun, 'orange', linewidth=2, label='Heun E(t)')
plt.plot(t_real, E_real, 'orange', linestyle='--', label='True E(t)')
plt.title("Exposed"); plt.xlabel("Time"); plt.ylabel("E")
plt.grid(True); plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, I_heun, 'r', linewidth=2, label='Heun I(t)')
plt.plot(t_real, I_real, 'r--', label='True I(t)')
plt.title("Infectious"); plt.xlabel("Time"); plt.ylabel("I")
plt.grid(True); plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, L_heun, 'green', linewidth=2, label='Heun L(t)')
plt.plot(t_real, L_real, 'green', linestyle='--', label='True L(t)')
plt.title("Latent"); plt.xlabel("Time"); plt.ylabel("L")
plt.grid(True); plt.legend()

plt.tight_layout()
plt.suptitle("TB Model: Heun's Method vs True Values", fontsize=14, y=1.02)
plt.show()

mse_S, mse_E, mse_I, mse_L = CalculateMSE(S_heun, E_heun, I_heun, L_heun)

print("\nMean Squared Errors (MSE) compared to Reference:")
print(f"S (Susceptible): {mse_S:.6f}")
print(f"E (Exposed)    : {mse_E:.6f}")
print(f"I (Infectious) : {mse_I:.6f}")
print(f"L (Latent)     : {mse_L:.6f}")

timesElapsed = timeit.repeat(
    lambda: heun_non_self_start(tb_model, y0, t, params),
    repeat=50,      # Run 10 separate trials
    number=1        # Each trial runs once (adjust for very fast functions)
)
timesElapsed = sorted(timesElapsed)
print('min Time', min(timesElapsed))
print('median Time', np.median(timesElapsed))
print('mean Time', np.mean(timesElapsed))
print('std Time', np.std(timesElapsed))
