# Liquid-Neural-Networks
Notebooks to train LTC, CTRNN, NODE and CTGRU on PTB Diagnostic and PTB-XL data.
# ECG Arrhythmia Classification with Continuous-Time Neural Networks

Comparative study of four continuous-time neural network architectures for binary ECG classification — **Liquid Time-Constant Network (LTC)**, **Continuous-Time RNN (CTRNN)**, **Neural ODE (NODE)**, and **Continuous-Time GRU (CT-GRU)** — evaluated across two benchmark datasets.

---

## Overview

Standard deep learning methods (CNNs, LSTMs) process ECG signals as discrete sequences with fixed time steps. Continuous-time neural networks instead model the hidden state as the solution to an ordinary differential equation (ODE), allowing the network to natively represent the temporal dynamics of cardiac signals such as RR intervals, QRS morphology, and ST-segment changes.

This repository benchmarks four such architectures side by side, using identical training pipelines, evaluation metrics, and data splits across both datasets.

**Task:** Binary classification — Healthy vs. Cardiac Disease (or Myocardial Infarction)  
**Framework:** PyTorch (pure, no external ODE solvers)  
**Metrics:** 12 per model — Accuracy, Sensitivity, Specificity, Precision, NPV, F1, F2, Balanced Accuracy, MCC, Cohen's Kappa, ROC-AUC, PR-AUC

---

## Datasets

### Notebook 1 — Pre-processed PTB Diagnostic ECG Database

| Property | Value |
|---|---|
| **Kaggle source** | `madhulikavikraman16/preprocessedzip` |
| **Path** | `PTB_1_STFT_files/*.npy` |
| **Patients** | 290 |
| **Recordings** | ~549 |
| **Input** | Single-lead isolated heartbeat waveform |
| **Timesteps** | 150 (fixed) |
| **Leads** | 1 |
| **Sampling rate** | 125 Hz (downsampled from 1000 Hz) |
| **Labels** | `0` = Control (healthy), `1` = Myocardial Infarction |
| **Split** | Subject-level 80/20 train/test |

**Preprocessing pipeline (applied prior to this repo):**

1. Resample 1000 Hz → 125 Hz
2. Split recording into 10-second segments
3. Detect R-peaks using `biosppy`
4. Extract one heartbeat centred on the median R-peak (1.2× RR window)
5. Min-max normalise each beat to [0, 1]
6. Pad or truncate to exactly 150 samples

Each `.npy` filename encodes the subject ID and label, e.g. `STFT_<subject_id>_..._control.npy` or `..._MI.npy`. The split is performed at the **subject level** to prevent data leakage — all beats from the same patient stay in the same partition.

---

### Notebook 2 — PTB-XL

| Property | Value |
|---|---|
| **Kaggle source** | `khyeh0719/ptb-xl-dataset` |
| **Patients** | ~18,900 |
| **Recordings** | ~21,800 |
| **Input** | 12-lead full ECG recording |
| **Timesteps** | 500 (downsampled 2× from 1000; original 100 Hz low-res) |
| **Leads** | 12 |
| **Sampling rate** | 50 Hz effective (after 2× downsample) |
| **Labels** | `0` = NORM only, `1` = confirmed cardiac disease |
| **Split** | Official stratified folds — folds 1–8 train, fold 9 val, fold 10 test |

**Label strategy:**

Records labelled `NORM` with no cardiac disease codes → **0 (healthy)**. Records containing any confirmed cardiac disease code (MI, STEMI, LBBB, AF, VT, HYP, ISCHAEMIA, etc.) → **1 (disease)**. Ambiguous records (neither purely NORM nor confirmed disease) are **dropped**. This avoids the common pitfall of treating all non-NORM records as positive.

**Normalisation:** Per-lead z-score (mean and std computed across the time axis, per recording per lead).

**Downsampling:** The low-resolution waveforms (`filename_lr`, 100 Hz, 1000 timesteps) are downsampled 2× via slice indexing (`x[::2, :]`) to 500 timesteps for computational feasibility. Clinical features (P-wave, QRS, T-wave) are fully preserved at 50 Hz.

---

## Models

All four models share the same classifier wrapper: a recurrent cell that processes one timestep at a time, followed by a dropout layer and a single linear output head producing raw logits. `BCEWithLogitsLoss` with `pos_weight` handles class imbalance.

---

### LTC — Liquid Time-Constant Network

The LTC models the membrane voltage of a neuron as an ODE where the time constant **τ is a function of both the input and the current state**:

```
τ(x, v) = τ_min + softplus(A·x_agg + B·v + b)

dv/dt = [ -v/τ + Σⱼ |wⱼ| · σ(σⱼ·(xⱼ − μⱼ)) · (Eⱼ − v) ] · |cm| / τ
```

Integrated with **Euler steps** (3 steps per timestep in the optimised version). Sensory gates are **precomputed for the full sequence** in one vectorised GPU call before the time loop, significantly reducing wall time.

**Learnable parameters include:** conductance weights `W`, reversal potentials `erev` (±1), synaptic midpoints `mu`, synaptic slopes `sigma`, leak voltage `vleak`, leak conductance `gleak`, membrane capacitance `cm`, and the three liquid time-constant parameters `A_tau`, `B_tau`, `b_tau`.

**Stability measures:**
- `clip_parameters()` called after every optimiser step to keep `cm`, `gleak`, `tau_min` in physically valid ranges
- Lower learning rate `3e-4` (vs `1e-3` for other models)
- Tighter gradient clip norm `0.5`
- NaN batch detection and skipping in the training loop

---

### CTRNN — Continuous-Time RNN

A structured ODE with a **leak term** that pulls the hidden state back toward zero:

```
dh/dt = -h/τ + tanh(W_in·x + W_rec·h + b)

h_{t+Δt} = h_t + Δt · dh/dt
```

`τ` is a **per-neuron fixed parameter** (not a function of input or state), kept positive via `softplus`. Integrated with Euler (2 steps per timestep).

---

### NODE — Neural ODE

The hidden state dynamics are defined by a **free-form neural network**:

```
dh/dt = f_θ(x, h) = tanh(W · [x ; h] + b)
```

Integrated with **RK4 (Runge-Kutta 4th order)** — 4 function evaluations per step, error O(dt⁵) vs Euler's O(dt²):

```
k1 = dt · f(x, h)
k2 = dt · f(x, h + k1/2)
k3 = dt · f(x, h + k2/2)
k4 = dt · f(x, h + k3)
h  = h + (k1 + 2·k2 + 2·k3 + k4) / 6
```

NODE has the **fewest parameters** of all four models (~1.5K for NB2).

---

### CT-GRU — Continuous-Time GRU

Maintains **M=4 parallel memory traces** at log-linearly spaced time scales. Each trace decays at a different rate, allowing the network to simultaneously represent fast (QRS-scale) and slow (ST-segment-scale) dynamics:

```
Time scales:  τ̃ᵢ = 10^(0.5·i)  →  [1, √10, 10, 10√10]
Decay:        decay_i = exp(-1 / τ̃ᵢ)

Per timestep (5 sub-steps):
  1. Retrieval:       r_ki  = softmax(-(W_r·[x;h] - ln_τ̃)²)
  2. Event detect:    q_k   = tanh(W_q · [x ; Σᵢ r_ki·h_hat[:,i]])
  3. Storage:         s_ki  = softmax(-(W_s·[x;h] - ln_τ̃)²)
  4. Decay update:    h_hat = ((1 - s_ki)·h_hat + s_ki·q_k) · decay
  5. Combine:         h_out = Σᵢ h_hat[:,:,i]
```

CT-GRU has the **most parameters** (~13K for NB2) due to the M-scale expansion of the weight matrices.

---

### Training configuration (shared across all models)

| Parameter | NB1 | NB2 |
|---|---|---|
| Optimiser | Adam | Adam |
| Learning rate | `1e-3` (LTC: `3e-4`) | `1e-3` (LTC: `3e-4`) |
| Weight decay | `1e-4` | `1e-4` |
| LR schedule | Cosine annealing | Cosine annealing |
| Gradient clip | `0.5` | `0.5` |
| Batch size | 32 | 64 |
| Epochs | 30 | 30 |
| Early stopping | Patience 7 (on test AUC) | Patience 7 (on val AUC) |
| Hidden size | 64 | 32 |
| Dropout | 0.3 | 0.3 |
| Loss | BCEWithLogitsLoss + pos_weight | BCEWithLogitsLoss + pos_weight |

---
