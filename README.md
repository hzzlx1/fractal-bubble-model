# Fractal Bubble Universe: A Topological Model for CMB and Galaxy Dynamics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20357850.svg)](https://doi.org/10.5281/zenodo.20357850)

This repository contains the complete codebase, data, and analysis scripts for the research presented in the paper **"Interpreting the CMB Large-Scale Anomalies: A Fractal Bubble Wall Model Integrated with Silk Damping"**.

The project proposes and tests a novel cosmological framework where observed phenomena like the Cosmic Microwave Background (CMB) and galaxy rotation curves are interpreted through the geometry of a **fractal bubble wall**, serving as a topological defect and holographic boundary.

## 🔬 Core Research Summary

This project implements a unified model that challenges standard ΛCDM cosmology by explaining key anomalies through geometry rather than unknown particles:

1.  **CMB Large-Scale Power Suppression**: The model naturally produces the observed low-multipole (ℓ < 30) power deficit in the CMB temperature spectrum. This is a geometric consequence of the fractal bubble wall acting as a reflective boundary.
2.  **Integration of Silk Damping**: The model incorporates the physically mandatory Silk damping effect from photon-baryon fluid diffusion. The damping scale `ℓ_D` is **not a free parameter** but is calculated from first principles using Planck 2018 cosmological parameters.
3.  **Parameter Locking & Economy**: The core parameter, the fractal dimension `D_f ≈ 1.774`, is **locked** based on independent analysis of Voyager 1 data from the heliopause. The model achieves competitive fits with very few free parameters.
4.  **Multispectra Validation**: The code performs joint analysis on simulated TT, TE, and EE CMB power spectra, demonstrating the model's consistency across temperature and polarization data.

## 📁 Repository Structure

The repository is organized by research domain for clarity:
fractal-bubble-model/
├── README.md # This file
├── code/ # All analysis scripts
│ ├── SPARC_Analysis/ # Galaxy Rotation Curve Analysis
│ │ ├── XA3.py # Generates visual comparisons (scatter plots, histograms, AICc) of SPARC geometry vs. MOND.
│ │ ├── R1.py # Independent test code for heliopause magnetic field anomalies (uses Voyager-1 inspired data).
│ │ └── BDV1.3.py # Tests the planetary magnetic field scaling law (B ∝ M^{-ω}) in log-space via linear regression.
│ └── CMB_Analysis/ # Cosmic Microwave Background Analysis
│ └── cmb+1.py # Main script for CMB multispectra validation. Implements the complete Fractal Bubble Wall Self-Echo model with Silk damping.
├── data/ # Processed observational datasets
│ ├── SPARC_Lelli2016c.mrt # SPARC galaxy rotation curve data
│ └── MassModels_Lelli2016c.mrt # Mass model data
└── Figures/ # Source code for generating key publication figures
└── ... # (e.g., power spectrum comparison, residual analysis plots)
## 🛠️ Setup and Dependencies

To run the scripts, you need **Python 3.8+** and the following libraries installed:
Core Scientific Stack
numpy
scipy
matplotlib
pandas
