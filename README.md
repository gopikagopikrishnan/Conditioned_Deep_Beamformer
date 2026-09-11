# Physics-Grounded Objective-Conditioned Deep Beamforming for Adaptive Resolution-Contrast Control in Plane-Wave Ultrasound Imaging

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![GPU Acceleration: CUDA](https://img.shields.io/badge/CUDA-%2376B900.svg?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

<div align="center">
  <img width="216" height="384" alt="Conditioned_BNN" src="https://github.com/user-attachments/assets/9b56e7f1-da9d-4d68-a461-f55cb703b718" />
  <br />
  <em>Figure 1: Imaging Objective Sweep Demo on PICMUS Experimental Resolution Data.</em>
</div>

## Overview
This repository contains the demo implementation for the paper **"Physics-Grounded Objective-Conditioned Deep Beamforming for Adaptive Resolution-Contrast Control in Plane-Wave Ultrasound Imaging"** (paper currently under review in *IEEE Transactions on Medical Imaging (TMI)*).

* **Core Mechanism:** A single neural network predicts per-element receive apodization from single-plane-wave channel data.
* **Dynamic Control:** Conditioned on an intent vector $\mathbf{q}$, users can select a resolution- or contrast-optimized reconstruction on the fly.
* **Physics-Grounded:** Each intent is supervised against a coherent plane-wave compounding (CPWC) target beamformed at the corresponding receive aperture, making intent a physically defined control rather than a post-processing visual style.
* **Explainable Pipeline:** The reconstruction framework retains a standard delay-and-sum structure where the network outputs dynamic weights $w$, and the final image is computed as:
  $$\sum_{e} w_e s_e$$

---

## Quick Start (Demo)
Run the self-contained demo notebook to see the model in action without downloading massive datasets:

* **File:** `tunability_demo.ipynb`
* **What it does:** Sweeps the intent knob $\mathbf{q} = [1-t, t]$ from $t=0$ to $t=1$ on the bundled PICMUS experimental resolution phantom (~3 MB).
* **Output:** Generates live plots of lateral Full Width at Half Maximum (FWHM) against the tuning parameter $t$.
* **Requirements:** No training or full dataset downloads required.
