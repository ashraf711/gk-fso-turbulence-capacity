# Generalized Kennedy Receiver Capacity Optimization under Atmospheric Turbulence

This repository investigates coherent-state binary phase-shift keying (BPSK) communication using Generalized Kennedy (GK) receivers over both deterministic and Gamma-Gamma atmospheric fading channels. The receiver displacement and source prior probabilities are jointly optimized to maximize the achievable mutual information under practical receiver constraints.

The work is motivated by free-space optical (FSO) communication systems operating in photon-starved regimes, where atmospheric turbulence significantly impacts the distinguishability of non-orthogonal coherent states and consequently degrades achievable information rates.

---

## Research Objectives

The primary goals of this project are:

* Evaluate achievable information rates of GK receivers under atmospheric turbulence.
* Compare deterministic and Gamma-Gamma fading channels.
* Jointly optimize receiver displacement and input symbol priors.
* Quantify the impact of turbulence on receiver operating parameters.
* Benchmark practical receiver performance against quantum limits.



---

## Project Structure

```text
src/
├── benchmarks.py          Quantum capacity benchmarks
├── entropy.py             Binary entropy utilities
├── io_utils.py            CSV export utilities
├── kennedy_channel.py     Mutual information models
├── optim_joint_adam.py    Joint (β,p) optimization
├── optim_p_newton.py      Prior optimization
├── plots.py               Publication-quality plotting
├── sweep.py               Main experiment driver
└── turbulence.py          Gamma-Gamma fading generation

scripts/
├── run_sweep.py           Main simulation script
└── regenerate_plots.py    Plot regeneration utility

outputs/
├── kennedy_capacity_results.csv
├── capacity_vs_N.pdf
├── optimal_beta_vs_N.pdf
└── optimal_prior_vs_N.pdf
```

---

## Installation

```bash
git clone git@github.com:ashraf711/gk-fso-turbulence-capacity.git
cd gk-fso-turbulence-capacity
pip install -r requirements.txt
```

---

## Running Experiments

From the project root:

```bash
PYTHONPATH=. python3 scripts/run_sweep.py
```

Generated results and figures are stored in:

```text
outputs/
```

---

## Results

The framework produces:

* Capacity versus photon number
* Optimal displacement versus photon number
* Optimal prior probability versus photon number


---


## Authors

### Md Ashraf Hossain Ifty
Department of Electrical and Computer Engineering  
The University of Alabama in Huntsville

### Rahul Kumar Bhadani
Department of Electrical and Computer Engineering  
The University of Alabama in Huntsville


