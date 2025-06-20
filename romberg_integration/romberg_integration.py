
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -------------------------
# TB Model Parameters
# -------------------------
params = {
    'Λ': 2.0,         # Recruitment rate
    'β': 0.025,       # Transmission rate
    'δ': 1.0,         # Infectivity of latent individuals
    'p': 0.3,         # Proportion progressing directly to infectious
    'μ': 0.0101,      # Natural death rate
    'k': 0.005,       # Progression rate from E to I
    'r₁': 0.0,        # Recovery from E (none here)
    'r₂': 0.8182,     # Recovery from I
    'φ': 0.02,        # Progression from I to L
    'γ': 0.01,        # Reactivation from L to I
    'd₁': 0.0227,     # TB death from I
    'd₂': 0.20        # TB death from L
}

# -------------------------
# Time discretization
# -------------------------
t0, t_end = 0.0, 20.0  # Time in years
N_steps = 40           # Number of time steps
ts = np.linspace(t0, t_end, N_steps + 1)
dt = ts[1] - ts[0]     # Time step size

# -------------------------
# Derivatives function
# -------------------------
def derivatives(t, y):
    """Compute the derivatives for all compartments"""
    s, e, i, l = y
    dS = params['Λ'] - params['β']*s*(i + params['δ']*l) - params['μ']*s
    dE = params['β']*(1 - params['p'])*s*(i + params['δ']*l) + params['r₂']*i - (params['μ'] + params['k']*(1 - params['r₁']))*e
    dI = params['β']*params['p']*s*(i + params['δ']*l) + params['k']*(1 - params['r₁'])*e + params['γ']*l - (params['μ'] + params['d₁'] + params['φ']*(1 - params['r₂']) + params['r₂'])*i
    dL = params['φ']*(1 - params['r₂'])*i - (params['μ'] + params['d₂'] + params['γ'])*l
    return np.array([dS, dE, dI, dL])

# -------------------------
# Romberg Integration
# -------------------------
def romberg_integrate(f, a, b, max_steps=5, tol=1e-6):
    """Romberg integration with adaptive error control"""
    table = np.zeros((max_steps, max_steps))
    h = b - a
    table[0, 0] = 0.5 * h * (f(a) + f(b))

    for i in range(1, max_steps):
        h /= 2
        total = sum(f(a + k*h) for k in range(1, 2**i, 2))
        table[i, 0] = 0.5 * table[i-1, 0] + h * total

        for j in range(1, i+1):
            table[i, j] = table[i, j-1] + (table[i, j-1] - table[i-1, j-1]) / (4**j - 1)

        if abs(table[i, i] - table[i-1, i-1]) < tol:
            break

    return table[i, i], i  # return integral and level of convergence

# -------------------------
# Solve TB Model Function
# -------------------------
def solve_tb_model(E0):
    """Solve the TB model using Romberg integration with error tracking"""
    S = np.zeros(N_steps + 1)
    E = np.zeros_like(S)
    I = np.zeros_like(S)
    L = np.zeros_like(S)

    S[0] = params['Λ'] / params['μ']  # Initial susceptible
    E[0] = E0
    I[0] = 0
    L[0] = 0

    errors = {'S': [], 'E': [], 'I': [], 'L': []}
    deriv_history = []

    for step in range(N_steps):
        t_a, t_b = ts[step], ts[step + 1]
        current_derivs = derivatives(t_a, [S[step], E[step], I[step], L[step]])
        deriv_history.append(current_derivs)

        # Linear interpolation for integrand functions
        def make_integrand(comp_index):
            def integrand(t):
                theta = (t - t_a) / dt
                s = S[step] * (1 - theta) + S[step] * theta
                e = E[step] * (1 - theta) + E[step] * theta
                i = I[step] * (1 - theta) + I[step] * theta
                l = L[step] * (1 - theta) + L[step] * theta
                return derivatives(t, [s, e, i, l])[comp_index]
            return integrand

        components = [S, E, I, L]
        names = ['S', 'E', 'I', 'L']

        for i, (var, name) in enumerate(zip(components, names)):
            integral, level = romberg_integrate(make_integrand(i), t_a, t_b)
            var[step + 1] = var[step] + integral

            # Estimate error using last few derivative magnitudes
            if len(deriv_history) > 2:
                deriv_scale = np.mean([abs(d[i]) for d in deriv_history[-3:]])
            else:
                deriv_scale = abs(current_derivs[i])

            error = (dt**5 / 2880) * deriv_scale / (4**level)
            errors[name].append(error)

    return S, E, I, L, errors

# -------------------------
# Calculate Relative Errors
# -------------------------
def calculate_relative_errors(approx, ref):
    """Calculate relative errors safely"""
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = np.abs(approx - ref) / np.where(np.abs(ref) > 0, np.abs(ref), 1)
    return np.nan_to_num(rel_error, nan=0.0)

def calculate_relative_error(true, approx):
  if(true == 0):
    return -1
  return abs(true - approx) / true

# -------------------------
# Run both scenarios
# -------------------------
print("Running TB model simulations...")
S0, E0, I0, L0, _ = solve_tb_model(E0=0)  # Disease-free
S1, E1, I1, L1, romberg_errors = solve_tb_model(E0=1)  # Endemic

ref_S, ref_E, ref_I, ref_L = generate_reference_solution()

# Calculate comparison errors
comp_errors_S = [calculate_relative_error(ref_S[i], S1[i]) for i in range(len(ref_S))]
comp_errors_E = [calculate_relative_error(ref_E[i], E1[i]) for i in range(len(ref_E))]
comp_errors_I = [calculate_relative_error(ref_I[i], I1[i]) for i in range(len(ref_I))]
comp_errors_L = [calculate_relative_error(ref_L[i], L1[i]) for i in range(len(ref_L))]

# -------------------------
# Prepare Theoretical Errors for each time point
# -------------------------
# Theoretical errors are calculated for each integration step (40 steps)
# We'll create arrays with 41 values (same length as ts) for plotting
# For t=0, we set theoretical error to 0
theo_err_S = np.zeros(len(ts))
theo_err_E = np.zeros(len(ts))
theo_err_I = np.zeros(len(ts))
theo_err_L = np.zeros(len(ts))

# Fill the arrays (the error at step i is associated with the value at i+1)
theo_err_S[1:] = romberg_errors['S']
theo_err_E[1:] = romberg_errors['E']
theo_err_I[1:] = romberg_errors['I']
theo_err_L[1:] = romberg_errors['L']

# Calculate relative errors in percentage
rel_err_S = comp_errors_S * 100
rel_err_E = comp_errors_E * 100
rel_err_I = comp_errors_I * 100
rel_err_L = comp_errors_L * 100

# -------------------------
# Print Error Comparison Table
# -------------------------
print("\n" + "="*100)
print("Error Analysis Summary for Tuberculosis Model (E(0)=1)")
print("="*100)
print(f"{'Variable':<10} {'Avg Theoretical Error':<25} {'Max Relative Error (%)':<25}")
print("-"*100)
print(f"{'S':<10} {np.mean(theo_err_S):<25.4e} {np.max(rel_err_S):<25.6f}")
print(f"{'E':<10} {np.mean(theo_err_E):<25.4e} {np.max(rel_err_E):<25.6f}")
print(f"{'I':<10} {np.mean(theo_err_I):<25.4e} {np.max(rel_err_I):<25.6f}")
print(f"{'L':<10} {np.mean(theo_err_L):<25.4e} {np.max(rel_err_L):<25.6f}")
print("="*100)

# -------------------------
# Print Complete Time Series Table (41 values)
# -------------------------
print("\n\n" + "="*150)
print("Complete Time Series Values for TB Model (E(0)=1)")
print("="*150)
#print(f"{'Time':<8} {'S(t)':<10} {'TheoErrS':<10} {'E(t)':<10} {'TheoErrE':<10} "
 #     f"{'I(t)':<12} {'TheoErrI':<12} {'L(t)':<12} {'TheoErrL':<12} "
  #    f"{'RelErrS(%)':<12} {'RelErrE(%)':<12} {'RelErrI(%)':<12} {'RelErrL(%)':<12}")
print(f"{'Time':<8} {'S(t)':<10} {'S ref':<10} {'E(t)':<10} {'E ref':<10} "
      f"{'I(t)':<12} {'I ref':<12} {'L(t)':<12} {'L ref':<12} "
      f"{'RelErrS(%)':<12} {'RelErrE(%)':<12} {'RelErrI(%)':<12} {'RelErrL(%)':<12}")
print("-"*150)

# Print all 41 values (from t=0 to t=20)
for i in range(len(ts)):
    print(f"{ts[i]:<8.1f} "
          f"{S1[i]:<10.4f} {ref_S[i]:<10.2e} "
          f"{E1[i]:<10.4f} {ref_E[i]:<10.2e} "
          f"{I1[i]:<12.6f} {ref_I[i]:<12.2e} "
          f"{L1[i]:<12.6f} {ref_L[i]:<12.2e} "
          f"{rel_err_S[i]:<12.6f} {rel_err_E[i]:<12.6f} "
          f"{rel_err_I[i]:<12.6f} {rel_err_L[i]:<12.6f}")

# -------------------------
# Plot Results
# -------------------------
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Tuberculosis Model Dynamics (E(0)=1)', fontsize=16)

axs[0,0].plot(ts, S1, 'b-', label='Romberg Solution')
axs[0,0].plot(ts, ref_S, 'r--', label='Reference Solution')
axs[0,0].set_title('Susceptible (S)')
axs[0,0].set_ylabel('Population')
axs[0,0].legend()
axs[0,0].grid(True)

axs[0,1].plot(ts, E1, 'g-', label='Romberg Solution')
axs[0,1].plot(ts, ref_E, 'r--', label='Reference Solution')
axs[0,1].set_title('Exposed (E)')
axs[0,1].legend()
axs[0,1].grid(True)

axs[1,0].plot(ts, I1, 'r-', label='Romberg Solution')
axs[1,0].plot(ts, ref_I, 'b--', label='Reference Solution')
axs[1,0].set_title('Infectious (I)')
axs[1,0].set_xlabel('Time (years)')
axs[1,0].set_ylabel('Population')
axs[1,0].legend()
axs[1,0].grid(True)

axs[1,1].plot(ts, L1, 'm-', label='Romberg Solution')
axs[1,1].plot(ts, ref_L, 'b--', label='Reference Solution')
axs[1,1].set_title('Lost to follow-up (L)')
axs[1,1].set_xlabel('Time (years)')
axs[1,1].legend()
axs[1,1].grid(True)

plt.tight_layout()
plt.savefig("tb_model_results.png", dpi=300)
plt.show()
