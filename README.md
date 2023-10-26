# Grey-Stellar-Atmosphere-Model
The 1-D, plane-parallel, local thermal equilibrium, grey stellar atmosphere model for an A0V Star

Final project for ASTR 545: Stellar Spectroscopy
@author: oanavesa
Created Thu Nov 1 18:25:20 2018
Modified October 26, 2023 for publishing on Github

An A0V star is a main-sequence A-type star that has an effective temperature of 9520 K and a surface gravity of log g = 4.14. It is composed of 70% Hydrogen, 29% Helium, and 1% Calcium.

This code creates 10 evenly spaced optical depths in log space, determines a temperature for each layer, solves the pressure gradient of each layer, and solves the detailed balancing equations.
The Schwarzschild Criterion to test for stability against convection was also implemented.

Uses Rosseland mean opacity tables from Lawrence Livermore National Laboratory's [OPAL code](https://opalopacity.llnl.gov/).
