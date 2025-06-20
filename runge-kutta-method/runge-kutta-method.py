import numpy as np
import matplotlib.pyplot as plt

# --- Parameters for the TB model ---
params =[
2,  # lambda (recruitment rate)
0.025,  # beta (transmission rate)
1,  # delta (differential infectivity)
0.3,  # p (fraction that goes directly to infectious)
0.0101,  # mu (natural death rate)
0.005,  # k (progression rate from exposed to infectious)
0,  # r1 (early treatment effectiveness, not used here)
0.8182,  # r2 (treatment rate of infectious)
0.02,  # phi (rate from I to L)
0.01,  # gamma (reactivation from L to I)
0.0227,  # d1 (death rate from I)
0.20  # d2 (death rate from L)
]

# --- Derivative function for TB model ---
def tb_derivatives(t, y, params):
    S, E, I, L = y
    λ, β, δ, p, μ, k, r1, r2, φ, γ, d1, d2 = params

    dSdt = λ - β * S * (I + δ * L) - μ * S
    dEdt = β * (1 - p) * S * (I + δ * L) + r2 * I - (μ + k * (1 - r1)) * E
    dIdt = β * p * S * (I + δ * L) + k * (1 - r1) * E + γ * L - (μ + d1 + φ * (1 - r2) + r2) * I
    dLdt = φ * (1 - r2) * I - (μ + d2 + γ) * L

    return np.array([dSdt, dEdt, dIdt, dLdt])


# --- RK4 integrator ---
def runge_kutta_4(f, y0, t, params):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(1, n):
        h = t[i] - t[i - 1]
        k1 = f(t[i - 1], y[i - 1], params)
        k2 = f(t[i - 1] + h / 2, y[i - 1] + h / 2 * k1, params)
        k3 = f(t[i - 1] + h / 2, y[i - 1] + h / 2 * k2, params)
        k4 = f(t[i - 1] + h, y[i - 1] + h * k3, params)
        y[i] = y[i - 1] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


# --- Choose scenario ---
ncase = 2  # Change to 1 for disease-free equilibrium

if ncase == 1:
    S0 = 2 / 0.0101  # ≈ 198.02
    E0 = 0
    I0 = 0
    L0 = 0
elif ncase == 2:
    S0 = 2 / 0.0101  # ≈ 198.02
    E0 = 1  # Introduce 1 exposed person
    I0 = 0
    L0 = 0

y0 = [S0, E0, I0, L0]

# --- Time vector ---
t = np.linspace(0, 20, 41)  # 41 steps → step size h = 0.5

# --- Solve the system using RK4 ---
solution = runge_kutta_4(tb_derivatives, y0, t, params)
S, E, I, L = solution.T  # transpose for plotting

# --- Plotting ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t, S, 'b', linewidth=2)
plt.title('S(t) - Susceptible')
plt.xlabel('Time (years)')
plt.ylabel('S')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, E, 'orange', linewidth=2)
plt.title('E(t) - Exposed')
plt.xlabel('Time (years)')
plt.ylabel('E')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, I, 'r', linewidth=2)
plt.title('I(t) - Infectious')
plt.xlabel('Time (years)')
plt.ylabel('I')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, L, 'green', linewidth=2)
plt.title('L(t) - Latent / Out of Sight')
plt.xlabel('Time (years)')
plt.ylabel('L')
plt.grid(True)

plt.tight_layout()
plt.suptitle('TB Model Simulation using Runge-Kutta 4th Order (ncase = {})'.format(ncase), fontsize=16, y=1.02)
plt.show()