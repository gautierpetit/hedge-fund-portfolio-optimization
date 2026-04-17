# Hedge Fund Portfolio Optimization: A Semi-Parametric Approach

This repository preserves the final report, selected outputs, and original academic implementation developed for my Master's thesis in Finance at HEC Lausanne, University of Lausanne.

The project studies hedge fund portfolio optimization under non-normal return distributions using a semi-parametric framework combining volatility modeling, tail modeling, and dependence modeling.

> **Note on code structure**  
> This repository preserves the original academic implementation used at the time of thesis submission. The code is shared primarily for research transparency and reproducibility and reflects the workflow used during the MSc thesis rather than my current software engineering standards.  
> For more recent examples of my current engineering approach, see my [MicroAlpha Engine](https://github.com/gautierpetit/microalpha-engine) and [Meta-Labeling Alpha Filter](https://github.com/gautierpetit/meta-labeling-alpha-filter) projects.

---

## Thesis Details

**Title:** Hedge Fund Portfolio Optimization: A Semi-Parametric Approach  
**Author:** Gautier Petit  
**Program:** Master of Science in Finance (MScF), HEC Lausanne, University of Lausanne  
**Date:** August 2024  
**Grade:** 6/6  
**Supervisor:** Prof. Thomas Cho, HEC Lausanne  
**Expert:** Prof. François-Serge Lhabitant, HKUST / Kedge Capital

**Thesis PDF:** [`thesis.pdf`](./thesis.pdf)

---

## Project Summary

Traditional portfolio optimization methods often struggle with hedge fund return data because such returns tend to exhibit skewness, fat tails, serial dependence, and other non-Gaussian features.

This thesis develops a semi-parametric allocation framework designed to better capture those characteristics by combining:

- AR(1)-EGARCH(1,1) modeling for conditional volatility
- Extreme Value Theory (EVT) with Generalized Pareto tails
- Student's t copula dependence modeling
- alternative portfolio objectives including CVaR, CDaR, and Omega ratio
- practical constraints such as turnover control and benchmark correlation limits

The framework is evaluated on more than 30 years of HFR strategy index data and compared against traditional benchmark allocations.

---

## Repository Purpose

This repository is intended primarily as a **research archive** rather than a polished software package.

It is meant to provide:

- the final thesis manuscript
- the original implementation used for the research
- saved figures and summary outputs
- transparency on methodology and results

The primary artifact is the thesis itself. The code is included as supporting material.

---

## Repository Structure

```text
├── data/                         # Input folder for licensed data (not distributed)
├── figures/                      # Output figures used in the thesis
│   ├── Correlation/
│   ├── Stackplots/
│   └── Turnover/
├── src_legacy/                   # Original academic implementation
│   ├── main.py
│   └── portfolios_functions.py
├── .gitignore
├── Indexes summary.xlsx          # Descriptive statistics and hypothesis test outputs
├── Performance Measures.xlsx     # Portfolio performance tables across scenarios
├── README.md
├── requirements.txt
└── thesis.pdf                    # Final defended thesis manuscript
```

---

## Methodology Overview

The thesis combines parametric and non-parametric components in a single portfolio construction framework:

1. Return modeling
   - descriptive statistics
   - normality, autocorrelation, and heteroskedasticity diagnostics
   - AR(1)-EGARCH(1,1) estimation
2. Semi-parametric distribution construction
   - EVT-based tail replacement using Generalized Pareto distributions
   - Gaussian smoothing of the center of the residual distribution
3. Dependence modeling
   - Student's t copula for cross-asset dependence
   - generation of synthetic return series preserving nonlinear dependence structure
4. Portfolio optimization
   - benchmark portfolios: Equal Weight, Minimum Variance, Maximum Sharpe
   - downside-risk portfolios using CVaR, CDaR, and Omega objectives
   - extensions with turnover penalties and benchmark correlation constraints

---

## Data Note

This repository does not distribute the original HFR data used in the thesis.

The project relies on proprietary input data, so full replication requires access to the relevant data sources. The thesis manuscript, figures, and saved output files are therefore the main materials intended for inspection.


---

## Reproducibility Note

The code in `src_legacy/` is the original thesis implementation preserved for transparency. It follows an academic research workflow and is not presented as a modern production-style package.

Readers interested primarily in the research contribution should start with:

- [`thesis.pdf`](./thesis.pdf)
- `Performance Measures.xlsx`
- `Indexes summary.xlsx`
- `the figures/ folder`

---

## License

Unless otherwise stated, the **BSD 3-Clause License** applies to source code in this repository only and does not apply to the thesis manuscript ([`thesis.pdf`](./thesis.pdf)), which remains © Gautier Petit, all rights reserved.

---

## Contact

**Gautier Petit** 

- [GitHub](https://github.com/gautierpetit): gautierpetit
- [LinkedIn](https://www.linkedin.com/in/gautierpetitch): gautierpetitch