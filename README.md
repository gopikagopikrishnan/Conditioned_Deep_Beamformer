# Physics-Grounded Objective-Conditioned Deep Beamforming for Adaptive Resolution-Contrast Control in Plane-Wave Ultrasound Imaging
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![GPU Acceleration: CUDA](https://img.shields.io/badge/CUDA-%2376B900.svg?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

<div align="center">
  <img width="216" height="384" alt="Conditioned_BNN" src="https://github.com/user-attachments/assets/9b56e7f1-da9d-4d68-a461-f55cb703b718" />
  <br />
  <em>Figure 1: Demonstration of the Conditioned BNN architecture.</em>
</div>

Intent-conditioned deep beamforming for plane-wave ultrasound. A single network predicts per-element receive apodization from single-plane-wave channel data, conditioned on an intent vector q that selects a resolution- or contrast-optimized reconstruction. Each intent is supervised against a compound (CPWC) target beamformed at the corresponding receive aperture, making intent a physically defined control rather than a post-processing style. The reconstruction remains delay-and-sum: the network outputs weights w, and the image is Σₑ wₑ sₑ.
Try it in 2 minutes: notebooks/tunability_demo.ipynb sweeps the intent knob q = [1−t, t] from t=0 to t=1 on the PICMUS experimental resolution phantom (bundled, ~3 MB) and plots lateral FWHM against t. No training, no full dataset download.
Code for the paper "Intent-Conditioned Deep Beamforming for Ultrasound: A Physically Grounded Resolution–Contrast Control" (submitted to IEEE TMI).

---

