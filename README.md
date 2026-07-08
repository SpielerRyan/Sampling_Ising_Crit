# Sampling the 2D Classical Ising Model at Criticality via Variational Autoregressive Networks (VAN)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Qaf2--FXYCVYkyKAuvuO7gC_nwR5Dy2E)

**[View the Fully Rendered Notebook on nbviewer](https://nbviewer.org/github/SpielerRyan/Sampling_Ising_Crit/blob/blob/main/Sampling_Ising.ipynb)**

Note that nbviewer has a GitHub API limit.  If clicking the above link gives an error stating that limit to have been exceed, use the open in Colab button or download the Sampling_Ising.ipynb and upload it to Colab to view.

Inspired by the framework in **arXiv:1809.10606** and the realization that the autoregressive property maps to a generalized **transfer matrix**, this repository utilizes a Pixel Convolutional Neural Network (PixelCNN) to sample configurations of the 2D classical Ising model at its critical temperature.

Using **PyTorch**, this project constructs an autoregressive network that explicitly respects physical priors and lattice symmetries, including scale invariance, reflection positivity, translation invariance, and the global $\mathbb{Z}_2$ spin-flip symmetry. The network's performance is rigorously benchmarked by examining the decay of spin-spin correlations, comparing thermodynamic observables against the exact **Onsager solution**, and computing the reverse Kullback-Leibler (KL) divergence.

---

## Architecture & Core Components

* **Ising Model Layer:** Implements the 2D Ising Model on a torus (periodic boundary conditions). It provides the framework for computing exact thermodynamic energy, magnetization, and spatial correlators used to evaluate the generative model.
* **Ising PixelCNN Architecture:** Defines the autoregressive sampling backbone. It incorporates custom **dilated masked convolutions** to enforce causality during the autoregressive generation chain while maximizing the receptive field to capture critical, long-range correlations.
* **Multi-Stage Training Pipeline:** Employs a dual-phase training strategy optimized via Adam. The structural limitations discovered in a baseline training run motivate a second, advanced phase incorporating **simulated annealing** within the variational loop to prevent mode collapse and force exploration of complex, low-energy configurations.
* **Evaluation:** Validates the model against known conformal field theory predictions by testing the spin-spin correlator for power-law decay ($1/r^{2\Delta}$ with conformal dimension $\Delta = 1/8$), benchmarking observables directly against Onsager's analytical limits, and tracking the reverse KL divergence to measure the distance between the neural network's distribution and the true canonical ensemble.

---

## How to Run

This project is completely self-contained and designed to run directly in the cloud via Google Colab. 

1. Click the **Open in Colab** badge at the top of this file.
2. **CRITICAL:** Ensure your runtime environment is configured to use a high-performance **GPU instance** (e.g., L4 or A100). Because of the deep autoregressive sampling chain, **training takes approximately 90 minutes on an L4 GPU.** Running on a standard CPU runtime will result in severe timeout errors.
3. Execute the cells from top to bottom.
