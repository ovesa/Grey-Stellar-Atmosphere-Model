# Grey-Stellar-Atmosphere-Model

A 1D, plane-parallel, gray stellar atmosphere model in Local Thermodynamic Equilibrium (LTE) for an A0V star, developed as a final project for ASTR 545: Stellar Spectroscopy.

The full report describing the methods and results is available in `Final_Report.pdf`.

---

## Overview

This code constructs a 10-layer atmosphere in log-spaced optical depth for an A0V main-sequence star (T_eff = 9520 K, log g = 4.14), composed of 70% Hydrogen, 29% Helium, and 1% Calcium. It solves for the temperature, pressure, opacity, and density at each layer, performs detailed balancing to compute ionization fractions and number densities for all species, and applies the Schwarzschild Criterion to test each layer's stability against convection.

The model is built under the assumption that the atmosphere is 1D, plane-parallel, and gray. This means that the Rosseland mean opacity is used and assumed to be independent of wavelength. LTE is assumed, allowing ionization states to be described by the Saha equation and excitation states by the Boltzmann equation.

---

## What the Code Computes

- Temperature at each atmospheric layer using the Hopf function
- Converged pressure at each layer via hydrostatic equilibrium
- Opacity at each layer from OPAL Table #72 (X = 0.70, Y = 0.29, Z = 0.01)
- Total mass density at each layer
- Electron density by solving the transcendental equation (via Brent's method)
- Electron, nuclear, gas, radiation, and total pressures, and their ratios
- Ionization fractions for H, He, and Ca via the Saha equation
- Number densities for all ionization stages of H, He, and Ca
- Hydrogen excitation density fractions for levels n = 1, 2, and 3
- Schwarzschild Criterion stability test for convection at each layer

---

## Key Results

The most physically interesting behavior occurs around an optical depth of τ ~ 10⁰–10¹, driven primarily by the onset of Hydrogen ionization:

- **Opacity** reaches a maximum value in this range before falling back, coinciding with the onset of Hydrogen ionization.
- **Total mass density** and **nuclear pressure** plateau in this same range, reflecting the pressure-density relationship under hydrostatic equilibrium.
- **Gas pressure** dominates the total pressure throughout the atmosphere. Radiation pressure is negligible in optically thin regions but grows rapidly in optically thick ones.
- The **electron-to-gas pressure ratio** (P_e / P_g) approaches 0.5 at τ ~ 10⁰, where Hydrogen is becoming fully ionized and Helium is beginning to ionize.
- **Calcium** is predominantly singly ionized throughout, with very little neutral Ca present. **Helium** remains mostly neutral until deeper layers. **Hydrogen** transitions from ~82% neutral at the surface to nearly fully ionized at depth.
- The **Schwarzschild Criterion** finds that layers 1–5 and layer 8 are stable against convection, while layers 6, 7, and 9 are unstable — consistent with the optically thick regions where Hydrogen is fully ionized.

---

## Methodology

### 1. Atmospheric layers
Ten layers are defined in log-spaced optical depth (τ = 10⁻³ to 10²). The temperature at each layer is computed from the Hopf function:

$$T(\tau) = T_{\rm eff} \left[\frac{3}{4}\left(\tau + q(\tau)\right)\right]^{1/4}$$

where q(τ) is interpolated from the Mihalas Hopf table.

### 2. Pressure gradient
The boundary pressure is estimated from Gray's figure 9.17 and scaled to the A0V surface gravity using:

$$\log P(\tau) = \log P'(\tau) + \frac{2}{3}(\log g - \log g')$$

Subsequent layers are solved iteratively using a convergence scheme that averages mean opacities between layers until the pressure satisfies hydrostatic equilibrium to within ε = 10⁻⁵.

### 3. Detailed balancing
At each layer, the electron density is solved from the charge conservation equation using Brent's method (`scipy.optimize.brentq`). The Saha equation gives ionization fractions for H, He, and Ca, and the Boltzmann equation gives Hydrogen excitation density fractions for n = 1, 2, 3.

### 4. Schwarzschild Criterion
Each layer is tested for convective stability by comparing the radiative and adiabatic temperature gradients:
- **Stable:** ∇_ad < r_d ∇_r
- **Unstable:** ∇_ad > r_d ∇_r


