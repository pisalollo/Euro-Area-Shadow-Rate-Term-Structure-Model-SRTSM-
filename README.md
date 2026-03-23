# Euro Area Shadow Rate Term Structure Model (SRTSM)

This repository contains a MATLAB implementation of a Shadow Rate Term Structure Model for the Euro Area. The codebase replicates and extends the framework proposed by Lemke & Vladu (2017) to estimate the latent shadow short rate and analyze monetary policy expectations during the Effective Lower Bound (ELB) period.

##  Project Overview
The model extracts the shadow rate by combining a baseline affine term structure model with a non-linear Extended Kalman Filter (EKF). It is designed to handle deep negative interest rates and structural shifts in the yield curve caused by unconventional monetary policies (like QE and forward guidance).

### Core Methodology
The architecture is built upon three foundational papers in term structure modeling:
1. **Joslin, Singleton, and Zhu (2011)**: Used for the baseline linear Affine Term Structure Model (ATSM) estimation and risk-neutral ($\mathbb{Q}$) parameter rotation.
2. **Lemke and Vladu (2017)**: The core shadow rate framework estimating a constant/dynamic Effective Lower Bound (ELB) and the associated shadow rate path.
3. **Wu and Xia (2016)**: Implemented their analytical approximation for bond pricing inside the non-linear measurement equation. This replaces computationally heavy Monte Carlo pricing, providing analytical Jacobians for the Extended Kalman Filter and ensuring extreme numerical stability.

## Current Features
- **PCA Factor Extraction**: Demeaning and extraction of Level, Slope, and Curvature factors from the pre-LB period.
- **Maximum Likelihood Estimation (MLE)**: Cross-sectional optimization of JSZ structural parameters.
- **Grid Search for ELB**: Automated grid search maximizing the log-likelihood of the EKF to estimate the market-perceived lower bound.
- **Out-of-Sample Forecasting (Zero Look-Ahead Bias)**: A dedicated module to compute the market-implied "Lift-off" timing (e.g., crossing the 25 bps threshold). The VAR ($\mathbb{P}$-dynamics) and ELB are re-estimated on strictly out-of-sample truncated data to simulate real-world trading desk forecasting.

## Data
Currently, the model runs on the **Euro Area AAA Government Yield Curve** (maturities: 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y). 
*Note: Because AAA bonds carry a strong convenience yield/scarcity premium due to ECB asset purchases, the estimated ELB naturally sits deeper (e.g., -80 bps) compared to OIS/EONIA curves.*

## Work In Progress / Future Steps
This is an active research project. Upcoming implementations include:
- [ ] **OIS/€STR Integration**: Replacing AAA yields with the EONIA/€STR swapped curve to strip out sovereign scarcity premia and exactly replicate Lemke-Vladu's dataset.
- [ ] **Time-Varying Lower Bound**: Linking the ELB dynamically to the ECB's Deposit Facility Rate (DFR) rather than estimating a single historical floor.
- [ ] **Term Premia Decomposition**: Extracting forward term premia decoupled from the ZLB asymmetry.

## How to Run
Run the main script `main.m`. The script is modularized into distinct blocks (Data Loading, JSZ Estimation, EKF Filtering, OOS Forecasting).