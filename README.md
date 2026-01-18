# # Data-Driven Discovery of Multi-Robot Dynamics

**Symbolic Equation Discovery & PINN-based Trajectory Reconstruction**

This repository contains a research pipeline for **discovering equations of motion of interacting robots from experimental data** and **reconstructing trajectories using Physics-Informed Neural Networks (PINNs)**.

The workflow combines:

* symbolic regression via **EPDE**,
* hyperparameter optimization with **Optuna**,
* and trajectory reconstruction using **TEDEOUS**.

The project is based on real experimental kinematic data of multiple robots.

---

## Repository Structure

```
.
├── solver.ipynb
├── discovery_science.ipynb
├── ode_external_tokens/
│   └── (additional force tokens for EPDE)
├── data_00_330_[30_bots_PWM_10_15cw_15ccw_D_41cm].MP4.pickle
├── output/
│   └── robot_{id}/
│       └── {n}_parts/
│           └── {with|without}_force/
│               ├── part_{k}_system_best_params.json
│               ├── part_{k}_system_history_plot.html
│               ├── part_{k}_system_importances_plots.html
│               ├── system_{k}.csv
│               └── (additional visualizations and logs)
└── README.md
```

---

##  Data Description

### Input data

* **Pickle file** contains raw experimental data extracted from video tracking.
* For each robot and timestep:

  * robot ID,
  * orientation angle (PWM, converted to radians),
  * 2D coordinates `(x, y)`.

### Preprocessing

Implemented in `DataProcessor`:

* extraction of coordinates and angles,
* normalization of spatial coordinates using:

  * Box–Cox transform,
  * MinMax scaling to `[0, 1]`.

---

## Equation Discovery Pipeline (`discovery_science.ipynb`)

### 1. Data segmentation

* Each robot trajectory is split into `n_parts`.
* Each part is processed independently to study temporal variability of dynamics.

### 2. Symbolic equation discovery

Performed using **EPDE**:

* variables: `x(t), y(t)`,
* derivatives up to second order,
* polynomial smoothing preprocessing,
* optional external tokens (PWM, interaction forces — currently disabled).

### 3. Hyperparameter optimization

For each trajectory segment:

* **Optuna** optimizes EPDE parameters:

  * polynomial window,
  * smoothing sigma,
  * boundary,
  * population size.
* Objective balances:

  * equation residuals,
  * solution complexity.

### 4. Equation post-processing

Discovered equations are:

* parsed and aligned term-wise,
* aggregated across solutions,
* converted into tabular form:

  * coefficients per term,
  * zero-filled for missing terms.

Saved as:

```
system_{part_id}.csv
```

---

## Output Description

For each robot / part configuration:

* `system_{k}.csv`
  Table of discovered equation terms and coefficients.
* `part_{k}_system_best_params.json`
  Best Optuna hyperparameters and objective score.
* Optuna visualizations:

  * optimization history,
  * parameter importances.
* EPDE solution visualizations.

Directory hierarchy encodes:

* robot ID,
* number of trajectory parts,
* presence or absence of external forces.

---

## PINN-based Trajectory Reconstruction (`solver.ipynb`)

Using discovered equations:

* coefficients are sampled (Monte Carlo),
* PINNs are trained with **TEDEOUS**,
* trajectories are reconstructed and compared to data.

This stage evaluates:

* robustness of discovered equations,
* sensitivity to coefficient uncertainty.

---

## How to Run

### Equation discovery

Open and run:

```text
discovery_science.ipynb
```

Key entry point:

```python
main(target_robot=78, num_parts=4, n_closest=0)
```

### Trajectory reconstruction

Open and run:

```text
solver.ipynb
```

---

## Dependencies

Core libraries:

* numpy, pandas, scipy
* matplotlib
* scikit-learn
* epde
* optuna
* tedeous
* dill

---

## Research Goals

* Discover interpretable equations of robot motion from experimental data.
* Analyze stability and consistency of symbolic models.
* Validate discovered dynamics via PINN-based trajectory reconstruction.
* Extend models with interaction forces and control signals.

---

## Notes

* External force and PWM derivative tokens are implemented but currently disabled.
* The repository is structured as a research codebase; refactoring into scripts is planned.
* Output folders may become large and should not be version-controlled.


