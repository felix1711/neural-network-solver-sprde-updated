# Unsupervised PINN Solver for Singularly Perturbed Differential Equations

PyTorch implementation of unsupervised physics-informed neural networks for two benchmark ODEs. Training uses only governing-equation residuals and boundary conditions. **Analytical solutions are used only after training for evaluation, never as training labels.**

## Test Problem 1 — Second-order reaction-diffusion equation

\[
-\varepsilon v''(u)+v(u)=\omega(u),\quad 0\leq u\leq1
\]
\[
\omega(u)=-\cos^2(\pi u)-2\varepsilon\pi^2\cos(2\pi u),
\quad v(0)=v(1)=0.
\]

The output \(v_\theta(u)=u(1-u)N_\theta(u)\) enforces both Dirichlet conditions exactly.

Analytical solution for post-training evaluation:
\[
v(u)=\frac{e^{-(1-u)/\sqrt{\varepsilon}}+e^{-u/\sqrt{\varepsilon}}}{1+e^{-1/\sqrt{\varepsilon}}}-\cos^2(\pi u).
\]

## Test Problem 2 — Third-order differential equation

\[
-\varepsilon v'''(u)+v(u)=\omega(u),\quad 0\leq u\leq1
\]
\[
\omega(u)=81\varepsilon^2\cos(3u)+3\varepsilon\sin(3u).
\]
\[
v(0)=0,\quad v(1)=3\varepsilon\sin(3),\quad v'(0)=9\varepsilon.
\]

The output transformation enforces the two value boundary conditions exactly. The derivative boundary condition is included as a penalty in the training loss.

Analytical solution for post-training evaluation:
\[
v(u)=3\varepsilon\sin(3u).
\]

## Method

- Fully connected network with three hidden layers and `tanh` activations.
- PyTorch automatic differentiation computes second- and third-order derivatives.
- Adam updates are interleaved with L-BFGS optimisation steps.
- Uniform collocation points on \([0,1]\).
- No analytical solution values are used during training.

The default stopping threshold is \(10^{-4}\) for the relevant training objective. This is a residual/total-loss threshold, not a solution-error threshold.

## Structure

```text
neural-network-solver-sprde/
├── README.md
├── requirements.txt
├── .gitignore
├── experiments/
│   ├── test_problem_1.py
│   └── test_problem_2.py
└── results/
    ├── test_problem_1/
    └── test_problem_2/
```

Each script generates solution comparison, absolute error and training loss plots, a loss-history CSV, metrics text file and model weights in its results folder.

## Install

Python 3.10+ recommended.

```bash
git clone https://github.com/YOUR_USERNAME/neural-network-solver-sprde.git
cd neural-network-solver-sprde
pip install -r requirements.txt
```

## Run

```bash
python experiments/test_problem_1.py
python experiments/test_problem_2.py
```

Example custom settings:

```bash
python experiments/test_problem_1.py --epsilon 0.001 --epochs 3000 --target-loss 1e-4
python experiments/test_problem_2.py --epsilon 0.01 --epochs 1500 --target-loss 1e-4
```

Use `--help` to see all parameters.

## Evaluation

After training, scripts report MAE, maximum absolute error, relative \(L_2\) error and final training objective. Analytical solutions are used only for post-training comparisons.

## Reproducibility

Random seeds and training parameters are configurable. Results may vary with software versions and hardware. Add full bibliographic references for the benchmark problems and numerical methods before publishing.
