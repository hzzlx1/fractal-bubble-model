# Fractal Bubble Universe: A Topological Model for Cosmic Microwave Background and Galaxy Rotation

This repository contains the Python codebase and data analysis scripts supporting the research on the "Fractal Bubble Wall Model" as an alternative explanation for cosmological anomalies. The model proposes that the observed Cosmic Microwave Background (CMB) and galaxy rotation curves can be explained by a fractal geometry of cosmic bubble walls.

## 🔬 Core Research Summary

This project demonstrates a novel approach to cosmology by unifying:
1.  **CMB Large-Scale Anomalies**: A fractal bubble wall structure naturally explains the low-multipole power suppression observed in Planck data.
2.  **Galaxy Rotation Curves**: The same fractal geometry provides a geometric interpretation for flat rotation curves without invoking dark matter.
3.  **Silk Damping Integration**: The model incorporates Silk damping physics to constrain the scale of the fractal structure, resulting in a parameter-economic fit to observational data.

## 📂 Repository Structure

The repository is organized by research domain:

*   `/SPARC_Analysis`
    *   Contains Python notebooks for fitting the **SPARC galaxy rotation curve dataset**.
    *   Compares the Fractal Bubble Model predictions against **MOND** (Modified Newtonian Dynamics).
*   `/CMB_Analysis` (NEW - Core Focus)
    *   Scripts for generating simulated Planck 2018 power spectra (**TT, TE, EE**).
    *   Includes the **Silk damping** filter implementation.
    *   Code for calculating **BIC** and **Bayesian Factor** comparisons against the standard ΛCDM model.
*   `/Data`
    *   Processed observational datasets (SPARC, Planck simulated maps).
*   `/Figures`
    *   Source code for generating key plots (e.g., Power Spectrum comparison, Residual analysis).

## 🛠️ Setup and Dependencies

To run the scripts, you need Python 3.8+ and the following libraries:
