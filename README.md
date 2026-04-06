# Gray Stellar Atmosphere Model

A self-contained numerical simulation of a 1D, plane-parallel stellar atmosphere in Local Thermodynamic Equilibrium (LTE), built for an A0V star. The model solves a coupled system of equations (hydrostatic equilibrium, radiative transfer, ionization balance, and convective stability) iteratively across 10 atmospheric layers using root-finding, table interpolation, and convergence schemes implemented in Python.

Developed as a final project for ASTR 545: Stellar Spectroscopy. The full report is available in `Final_Report.pdf`.

---

## Technical Skills

- **Numerical methods:** iterative pressure convergence, Brent's method root-finding (`scipy.optimize.brentq`), 2D table interpolation (`scipy.interpolate`)
- **Mathematical modeling:** translating a coupled system of physics equations into modular, working code
- **Scientific computing:** `numpy`, `scipy`, `matplotlib`
- **Code design:** physics functions cleanly separated into `datafunctions.py`, keeping the main script readable and the library reusable
- **Validation:** convergence tolerance checking (ε = 10⁻⁵), Schwarzschild stability criterion as a physical self-consistency check

---

## What the Model Solves

Starting from a boundary condition at the top of the atmosphere, the model iterates layer by layer through a coupled system:

```
For each atmospheric layer:
    ├── Temperature          ← Hopf function + optical depth grid
    ├── Pressure             ← hydrostatic equilibrium (iterative convergence)
    ├── Opacity              ← 2D interpolation of OPAL Table #72
    ├── Total density        ← equation of state
    ├── Electron density     ← charge conservation (Brent's root-finding)
    ├── Ionization fractions ← Saha equation (H, He, Ca)
    ├── Number densities     ← all ionization stages
    ├── Excitation fractions ← Boltzmann equation (H, n=1,2,3)
    └── Convective stability ← Schwarzschild Criterion
```

All quantities are self-consistently coupled: pressure depends on opacity, opacity depends on density, density depends on electron density, and electron density depends on the ionization balance, which in turn depends on temperature and pressure.

---

## Star Modeled

| Parameter | Value |
|---|---|
| Effective temperature T_eff | 9520 K |
| Surface gravity log g | 4.14 |
| Hydrogen mass fraction X | 0.70 |
| Helium mass fraction Y | 0.29 |
| Calcium mass fraction Z | 0.01 |
| Opacity table | OPAL Table #72 |

---

## Key Results

The most physically interesting behavior occurs around τ ~ 10⁰–10¹, driven by the onset of Hydrogen ionization:

- **Opacity** peaks in this range, coinciding with Hydrogen becoming ionized.
- **Gas pressure** dominates throughout; radiation pressure grows rapidly only in optically thick regions.
- The **electron-to-gas pressure ratio** approaches 0.5 at τ ~ 10⁰, where Hydrogen is fully ionizing, and Helium begins to ionize.
- **Hydrogen** transitions from ~82% neutral at the surface to nearly fully ionized at depth; **Calcium** is predominantly singly ionized throughout.
- The **Schwarzschild Criterion** identifies the outermost layers as unstable to convection, which is consistent with the optically thick regime where Hydrogen ionization drives the instability.

---
