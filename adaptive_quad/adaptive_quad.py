
import numpy as np
import matplotlib.pyplot as plt

current_iter = 0
steps = 0
# Parameters
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

# ODE system
def f(t, y, p):
    S, E, I, L = y
    dS = p['lambda_'] - p['beta'] * S * (I + p['delta'] * L) - p['mu'] * S
    dE = p['beta'] * (1 - p['p']) * S * (I + p['delta'] * L) + p['r2'] * I - (p['mu'] + p['k'] * (1 - p['r1'])) * E
    dI = p['beta'] * p['p'] * S * (I + p['delta'] * L) + p['k'] * (1 - p['r1']) * E + p['gamma'] * L - \
         (p['mu'] + p['d1'] + p['phi'] * (1 - p['r2']) + p['r2']) * I
    dL = p['phi'] * (1 - p['r2']) * I - (p['mu'] + p['d2'] + p['gamma']) * L
    return np.array([dS, dE, dI, dL])

# Adaptive integration
def adaptive_integrate(f, y0, t0, t_end, params, h0=0.01, tol=1e-3, max_iter = 100000):
    global current_iter, steps
    t = t0
    y = y0.copy()
    h = h0
    t_vals = [t]
    y_vals = [y0.copy()]

    # Dormand-Prince coefficients from table2 here:
    # https://www.sciencedirect.com/science/article/pii/0771050X80900133?via%3Dihub

    c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1], dtype=float) #time increment

    # coefficients for rk5
    b_hat = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=float)

    # coefficients fo rk4
    b = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=float)

    # coeffcients for k_values
    A = np.array([
        [0,            0,            0,            0,           0,            0,     0],
        [1/5,          0,            0,            0,           0,            0,     0],
        [3/40,         9/40,         0,            0,           0,            0,     0],
        [44/45,        -56/15,       32/9,         0,           0,            0,     0],
        [19372/6561,   -25360/6561,  64448/6561,   -212/6561,   0,            0,     0],
        [9017/3168,    -355/33,      46732/5247,   49/176,      -5103/18656,  0,     0],
        [35/384,       0,            500/1113,     125/192,     -2187/6784,   11/84, 0]
    ], dtype=float)

    # Saftey factors for step size control
    safety_factor = 0.9
    max_h_increase = 5.0
    min_h_decrease = 0.2

    k_values = [np.zeros_like(y) for _ in range(len(c))]
    #current_iter = 0
    while t < t_end and current_iter < max_iter:
        current_iter += 1
        # Adjust h to not overshoot t_end
        if t + h > t_end:
            h = t_end - t
            if h < 1e-12: # Avoid tiny steps at the very end
                break

        k_values[0] = f(t, y, params)

        for i in range(1, len(c)):
            sum_terms = np.zeros_like(y)
            for j in range(i):
                sum_terms += A[i, j] * k_values[j]
            k_values[i] = f(t + c[i]*h, y + h * sum_terms, params)

        k_array = np.array(k_values)
        y_rk4 = y + h*np.sum(b[:, np.newaxis] * k_array, axis=0)
        y_rk5 = y + h*np.sum(b_hat[:, np.newaxis] * k_array, axis=0)

        # Estimate local error
        err = np.linalg.norm(y_rk5 - y_rk4, ord=np.inf)

        # Adjust step size
        if err == 0:
            h_new = h * max_h_increase
        else:
            h_new = h * safety_factor * (tol/err)**(1/5)

        h_new = np.clip(h_new, h * min_h_decrease, h * max_h_increase)

        if err <= tol:
            steps += 1
            t += h
            y = y_rk5
            t_vals.append(t)
            y_vals.append(y.copy())
            h = h_new
        else:
            h = h_new
            if h < 1e-15:
              break

    return np.array(t_vals), np.array(y_vals)

# Initial conditions
S0 = params['lambda_'] / params['mu']
y0 = np.array([S0, 1, 0, 0], dtype=float)


t_vals, y_vals = adaptive_integrate(f, y0, 0, 20, params, h0=0.01, tol=1e-7)
df_full = pd.DataFrame({
    'Time': t_vals,
    'S': y_vals[:, 0],
    'E': y_vals[:, 1],
    'I': y_vals[:, 2],
    'L': y_vals[:, 3]
})
# Plot results
labels = ['S(t)', 'E(t)', 'I(t)', 'L(t)']
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

df_sol = generate_reference_solution()
time_true = df_sol['Time'].values.astype(np.float32)
S_true = df_sol['S'].values
E_true = df_sol['E'].values
I_true = df_sol['I'].values
L_true = df_sol['L'].values
solutions_true = [S_true, E_true, I_true, L_true]
for i, ax in enumerate(axs.flat):
    ax.plot(t_vals, y_vals[:, i], label=labels[i], color='C'+str(i), linewidth=2)
    ax.plot(time_true, solutions_true[i], color='black', linestyle='--', linewidth=1, label='S - Reference')
    ax.set_title(labels[i])
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Population")
    ax.grid(True)

plt.tight_layout()
plt.suptitle("TB ODE Model - Adaptive Integration Results", fontsize=14, y=1.03)
plt.show()

print(f"Iterations: {current_iter}")
print(f"Steps: {steps}")

# Calculate Mean Squared Error for each compartment
mse_S, mse_E, mse_I, mse_L = CalculateMSEDifferentTimes(y_vals[:, 0], y_vals[:, 1], y_vals[:, 2], y_vals[:, 3], t_vals)

print("\nMean Squared Errors (MSE) compared to Reference:")
print(f"S (Susceptible): {mse_S:.6f}")
print(f"E (Exposed)    : {mse_E:.6f}")
print(f"I (Infectious) : {mse_I:.6f}")
print(f"L (Latent)     : {mse_L:.6f}")

timesElapsed = timeit.repeat(
    lambda: adaptive_integrate(f, y0, 0, 20, params, h0=0.01, tol=1e-7),
    repeat=50,      # Run 10 separate trials
    number=1        # Each trial runs once (adjust for very fast functions)
)
timesElapsed = sorted(timesElapsed)
print('min Time', min(timesElapsed))
print('median Time', np.median(timesElapsed))
print('mean Time', np.mean(timesElapsed))
print('std Time', np.std(timesElapsed))
