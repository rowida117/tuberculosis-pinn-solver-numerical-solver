# tuberculosis-pinn-solver-numerical-solver
#  Physics-Informed Neural Network for Tuberculosis Modeling
# Tuberculosis PINN Solver

Welcome! This project explores the use of Physics-Informed Neural Networks to solve a tuberculosis ODE model.

![Results](figures/pinn_vs_numerical.png)

📄 [Download Full Report](report.pdf)

This project presents a computational solution to a tuberculosis (TB) compartmental model using:
- Classical numerical solvers
- Physics-Informed Neural Networks (PINNs)

The PINN approach integrates the system of ODEs into the neural network training process, allowing for continuous and accurate approximations of disease dynamics over time.

---

## Project Overview

We modeled the spread and progression of TB using a system of four ODEs representing the following compartments:
- **S(t):** Susceptible
- **E(t):** Exposed
- **I(t):** Infected
- **L(t):** Latent

Two solution methods were implemented:
1. **Numerical integration methods** (e.g., Runge-Kutta-Fehlberg, Heun’s Method, Gauss Quadrature)
2. **Physics-Informed Neural Network (PINN)** trained using PyTorch

---

##  Methodology

- PINN implementation includes:
  - Fully-connected feedforward network
  - Tanh and Softplus activations
  - Loss terms: physics residual, initial conditions, initial derivative, data points
  - Two-phase optimization: Adam + L-BFGS
  - Curriculum learning for gradual physics emphasis

- Numerical results were used both for evaluation and for guiding PINN convergence.

---

## Results

- PINN successfully approximated all compartments and extrapolated 5× beyond training range.
- Final extrapolated mean square errors (MSE):

| Compartment | MSE (t = 0–100) |
|-------------|-----------------|
| S           | 0.281159        |
| E           | 0.264011        |
| I           | 0.002425        |
| L           | 0.000004        |

- PINN outperformed numerical integration in speed (7900% faster).

![PINN vs Numerical](figures/pinn_vs_numerical.png)

---

##  Files

| File | Description |
|------|-------------|
| `main.py` or `notebook.ipynb` | Full PINN implementation |
| `classical_methods.py`        | RKF, Heun, and quadrature solvers |
| `report.pdf`                  | Full report for course submission |
| `figures/`                    | Plots and comparison results |
| `index.md`                    | Webpage for GitHub Pages |

---

##  [View the GitHub Page](https://rowida117.github.io/tuberculosis-pinn-solver-numerical-solver/)

Live interactive summary hosted with GitHub Pages.

---

## 👥 Team Members

- [rowida moahemd ]  
- [zyad hamed ]  
- [Ali ahmed gad ]
- [yousef samy ]
- [yousef mortada ]
- [Mohamed ward ]
- [malak sherif ]
- [yomna sabry ]
- [mohamed nasser ]
- [ Abdelrahman Emad ]
- [farah yehia ]
- [yousef magdy]
  

---

##  References

- Raissi, M., Perdikaris, P., & Karniadakis, G. (2019). Physics-Informed Neural Networks. JCP.
- Lu et al., DeepXDE: A deep learning library for solving differential equations.
- Cairo University Numerical Methods materials (Spring 2025)

---

##  Contact

rowida.mohamed04@eng-st.cu.edu.eg

