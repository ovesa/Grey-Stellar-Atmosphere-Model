#!/usr/bin/env python
# coding: utf-8

# @author: oanavesa
# Created Thu Nov 1 18:25:20 2018
# Modified October 26, 2023 for publishing on Github

# This code is a final project for ASTR 545: Stellar Spectroscopy
# to model a 1-D, plane parallel, gray stellar atmosphere model in local thermal equilibrium
# of an A0V star.
# An A0V star is a main-sequence A-type star that has an effective temperature of 9520 K
# and a surface gravity of log g = 4.14.
# It is composed of 70% Hydrogen, 29% Helium, and 1% Calcium.

# This code creates 10 evenly spaced optical depths in log space,
# determines a temperature for each layer, solves for the pressure gradient of each layer,
# and solves the detailed balancing.
# The Schwarzschild Criterion to test for stability against convection was also coded.

############################################################################################
################################ Import Necessary Libraries ################################
############################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-ticks")
import scipy
from scipy import optimize
import os
import datafunctions
from tqdm import tqdm

##########################################################################################
################################ Read in Necessary Tables ################################
##########################################################################################

# Paths to files
current_directory = os.getcwd()
path_to_tables = current_directory + "/Tables/"
path_to_save_figures = current_directory + "/Figures/"

# Table names
hydrogen_partition_functions = "Partition_functions_for_Hydrogen.txt"
Opal_table_unpacked = "OPAL_table72.txt"
Opal_table_unpacked_no_Ts = "OPAL_table72_noT.txt"
Hopf_table = "hopf.txt"
partition_functions = "partition_functions.txt"

# Partition function for the Hydrogen excitation density fractions
hydrogen_partition_values = np.loadtxt(path_to_tables + hydrogen_partition_functions)

# Partition function
upot_values = np.loadtxt(path_to_tables + partition_functions, unpack=True)

# Hopf Table for Temperature-Tau relationship
taup, hopf_values = np.loadtxt(path_to_tables + Hopf_table, skiprows=1, unpack=True)

# OPAL TABLE #72; X = 0.7; Y = 0.29; Z = 0.01
opal_logT = np.loadtxt(path_to_tables + Opal_table_unpacked, usecols=[0])
OPAL_logR = np.genfromtxt(path_to_tables + Opal_table_unpacked, unpack=True)
OPAL_nologT = np.loadtxt(path_to_tables + Opal_table_unpacked_no_Ts)

#####################################################################################################
################################ Define Constants for the Atmosphere ################################
#####################################################################################################

# Theta values for the partition function
theta_partition_function_values = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    9999,
]

# Temperatures for the partition function
temperature_partition_function_values = [
    5040 / 0.2,
    5040 / 0.4,
    5040 / 0.6,
    5040 / 0.8,
    5040 / 1.0,
    5040 / 1.2,
    5040 / 1.4,
    5040 / 1.6,
    5040 / 1.8,
    5040 / 2.0,
    5040 / 9999,
]

# Columns = [H-,H,He,He+,Ca,Ca+]
chi_partition_function_values = [
    13.5984340 * 1.6022e-12,
    24.5873876 * 1.6022e-12,
    54.417763 * 1.6022e-12,
    6.113158 * 1.6022e-12,
    11.87172 * 1.6022e-12,
]

# Opacity values from OPAL Table #72
opal_table72_opacities = [
    -8.0,
    -7.5,
    -7.0,
    -6.5,
    -6.0,
    -5.5,
    -5.0,
    -4.5,
    -4.0,
    -3.5,
    -3.0,
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
]

# Create 10 evenly spaced layers in log space for the optical depth
log_tau_values = np.logspace(-3, 2, 10)

# A0V star composition
X = 0.7  # Mass fraction of Hydrogen (H)
Y = 0.29  # Mass fraction of Helium (He)
Z = 0.01  # Mass fraction of some metal; choose any one (Calcium / Ca, in this case)

A_H = 1.00794e0  # Atomic weight of Hydrogen
A_He = 4.002602e0  # Atomic weight of Helium
A_Ca = 40.078e0  # Atomic weight of Calcium

# Effective temperature [K]
T_eff = 9520  # A0V star

# Surface gravity
surface_gravity_g = 4.14  # log
log_surface_gravity_g = 10 ** (4.14)

# Boundary condition
# Boundary pressure not scaled - from Gray Plot found in notes
P1 = 1.8  # [dynes/cm^2] for log_g(4.0)

# Constants
k = 1.380658e-16  # Boltzmann Constant [cm^2 g s^-2 K^-1]
ma = 1.66054e-24  # Atomic weight mass to g
m_h = 1.6737236 * 10 ** (-24)  # Mass of Hydrogen [g]
m_he = 6.6464764 * 10 ** (-24)  # Mass of Helium, [g]
m_ca = 6.6551079 * 10 ** (-23)  # Mass of Calcium [g]
me = 9.10938 * 10 ** (-28)  # Mass of an electron [g]
C_phi = 4.83 * 10 ** (15.0)  # For the partition function
a = 7.56 * 10 ** (-15)  # Constant [erg cm^-3 K^-4] used for the radiation pressure
R_1 = 13.598 * 1.60218e-12  # Constant [ergs] used for the partition function

# Length of the opacity grid
length_opacity_grid = np.linspace(-8.0, 1.0, 70)

# If file exists, open it
if os.path.isfile(path_to_tables + "interpolatedRs.txt"):
    interpolated_opacities = np.loadtxt(
        path_to_tables + "interpolatedRs.txt",
        unpack=True,
    )
else:
    # Newly interpolated OPAL Table #72; size = 70 x 70
    interpolated_opacities = datafunctions.interpolate_opacities(
        length_opacity_grid, opal_table72_opacities, OPAL_nologT, save_table=False
    )


###################################################################################################
################################ Define Atmospheric Boundary Layer ################################
###################################################################################################

# Define the boundary pressure (log)
boundary_pressure = datafunctions.pressure_surface_gravity_scale(
    P1, surface_gravity_g, 4.0
)

# Un-log the boundary_pressure
boundary_pressure = 10 ** (boundary_pressure)

# Interpolate the Hopf Function, q(tau), for any tau value
interpolated_tau_values = scipy.interpolate.interp1d(taup, hopf_values)

temperature_boundary_layer = datafunctions.hopf_function(
    T_eff, log_tau_values[0], interpolated_tau_values(log_tau_values[0])
)
# Temperature array in log form
temperature_boundary_layer_log = np.log10(temperature_boundary_layer)

# Temperatures are interpolated and will give a value in the corresponding column of LogR[column] from OPAL Table #72
interpolated_temperature_values = scipy.interpolate.interp1d(opal_logT, OPAL_logR[1])

# Interpolates the log R's from OPAL Table #72
R_interpolate = scipy.interpolate.interp1d(opal_table72_opacities, OPAL_nologT[0])

# Calculates the mean molecular weight for all nuclear particles
mu_N_value = datafunctions.mu_n(X, A_H, Y, A_He, Z, A_Ca)

# Interpolate the partition function for the Hydrogen excitation density fractions
interpolated_H_partition_function = scipy.interpolate.interp1d(
    theta_partition_function_values,
    hydrogen_partition_values,
    bounds_error=False,
    fill_value="extrapolate",
)

# Abundance fractions of Hydrogen, Helium, and Calcium
abundance_H = datafunctions.abundance_fraction(X, A_H, Y, A_He, Z, A_Ca)
abundance_He = datafunctions.abundance_fraction(Y, A_He, X, A_H, Z, A_Ca)
abundance_Ca = datafunctions.abundance_fraction(Z, A_Ca, X, A_H, Y, A_He)

# Nuclear density of all atomic particles
n_nuclear = datafunctions.solve_for_nuclear_particle_density(
    boundary_pressure, k, temperature_boundary_layer
)

# Electron density at the boundary layer
electron_density_boundary_layer = optimize.brentq(
    datafunctions.transcential_eq,
    -1e5,
    4e22,
    xtol=10**-5,
    args=(
        boundary_pressure,
        temperature_boundary_layer,
        k,
        C_phi,
        upot_values,
        chi_partition_function_values,
        abundance_H,
        abundance_He,
        abundance_Ca,
        temperature_partition_function_values,
    ),
)

# Number density of Hydrogen, Helium, and Calcium.
n_H = datafunctions.solve_for_n_k(abundance_H, n_nuclear)
n_He = datafunctions.solve_for_n_k(abundance_He, n_nuclear)
n_Ca = datafunctions.solve_for_n_k(abundance_Ca, n_nuclear)

# Total density at boundary layer
rho_boundary_layer = datafunctions.rho_total(
    n_H, X, n_He, Y, n_Ca, Z, electron_density_boundary_layer, ma, me
)

# Compute the opacity at boundary layer
opacity_boundary_layer = datafunctions.obtain_chi_value(
    temperature_boundary_layer,
    opal_logT,
    interpolated_opacities,
    temperature_boundary_layer_log,
    rho_boundary_layer,
    length_opacity_grid,
)


# Opacities are logged; therefore, have to un-log them
opacity_boundary_layer = 10**opacity_boundary_layer

#######################################################################################
################################ Defining Atmospheric Layers ##########################
#######################################################################################

# Define all parameters and layers

# Temperatures at each layer
temperature_layers = []
# Pressures at each layer
pressure_layers_list = []
# Opacities at each layer
opacity_layers_list = []
# Total densities per layer
rho_layers_list = []
# Electron number density per layer
electron_number_density_list = []

# Pressures lists
electron_pressures_list = []
nuclear_pressures_list = []
gas_pressure_list = []
radiation_pressure_list = []
total_pressure_list2 = []
beta_list = []

# ionization fraction for hydrogen list
f11_hydrogen_list = []
f21_hydrogen_list = []

# number density of Hydrogen list
n_H_list = []
n_list = []
n11_hydrogen_list = []
n21_hydrogen_list = []

# ionization fraction for helium list
f12_helium_list = []
f22_helium_list = []
f32_helium_list = []

n12_helium_list = []
n22_helium_list = []
n32_helium_list = []

# ionization fraction for calcium list
f13_calcium_list = []
f23_calcium_list = []
f33_calcium_list = []
n13_calcium_list = []
n23_calcium_list = []
n33_calcium_list = []

# optical depth lists
phys_vs_opt_list = []

# Hydrogen excitation density fractions lists
Hydrogen_excitation_values_n1_list = []
Hydrogen_excitation_values_n2_list = []
Hydrogen_excitation_values_n3_list = []

# Scans through all of the defined tau values
# Compute the Hopf Function to calculate the temperature for each layer in the atmosphere as a function of optical depth
for i in tqdm(range(0, len(log_tau_values)), desc="Defining Temperature Layers"):
    temperature_at_layer_i = datafunctions.hopf_function(
        T_eff, log_tau_values[i], interpolated_tau_values(log_tau_values[i])
    )
    temperature_layers.append(temperature_at_layer_i)

# Temperature array in log form
temperature_layers_log = np.log10(temperature_layers)


# Append any boundary values calculated aboves
opacity_layers_list.append(opacity_boundary_layer)
pressure_layers_list.append(boundary_pressure)
phys_vs_opt_list.append(0)


# Loop starts atmosphere creation


# IndexError given of index 10 is out of bounds for axis 0 with size 10
# However, it still works
# Abundances are constant
try:
    for i in tqdm(
        range(0, len(log_tau_values)),
        desc="Calculating electron density and density per layer",
    ):
        electron_density_at_layer_i = optimize.brentq(
            datafunctions.transcential_eq,
            -1e5,
            4e30,
            xtol=0.00001,
            args=(
                pressure_layers_list[i],
                temperature_layers[i],
                k,
                C_phi,
                upot_values,
                chi_partition_function_values,
                abundance_H,
                abundance_He,
                abundance_Ca,
                temperature_partition_function_values,
            ),
        )

        electron_number_density_list.append(electron_density_at_layer_i)

        rho_at_layer_i = datafunctions.obtain_rho(
            electron_number_density_list[i],
            pressure_layers_list[i],
            temperature_layers[i],
            k,
            X,
            A_H,
            Y,
            A_He,
            Z,
            A_Ca,
            ma,
            me,
        )

        rho_layers_list.append(rho_at_layer_i)

        datafunctions.convergence(
            pressure_layers_list[i],
            i,
            i + 1,
            surface_gravity_g,
            log_tau_values,
            opacity_layers_list,
            temperature_layers,
            temperature_layers_log,
            rho_layers_list,
            pressure_layers_list,
            opal_logT,
            interpolated_opacities,
            length_opacity_grid,
        )
except IndexError as e:
    print(f"{e}")


# Solves for all of the other detailed balancing in the star using the information from the loop above. Must run that one first.

# IndexError given of  index 10 is out of bounds for axis 0 with size 10
# However, it still works

try:
    for i in tqdm(range(0, len(electron_number_density_list)), desc="Create Pressures"):
        # electron pressures
        electron_pressure_values = datafunctions.electron_pressure(
            k, temperature_layers[i], electron_number_density_list[i]
        )
        electron_pressures_list.append(electron_pressure_values)

        # radiation pressures
        radiation_pressure_value = datafunctions.radiation_pressure(
            temperature_layers[i], a
        )
        radiation_pressure_list.append(radiation_pressure_value)

        # nuclear pressures
        nuclear_pressure_values = datafunctions.nuclear_pressure2(
            rho_layers_list[i], temperature_layers[i], mu_N_value, k, ma
        )
        nuclear_pressures_list.append(nuclear_pressure_values)

        # gas pressures
        gas_pressure_value = nuclear_pressure_values + electron_pressure_values
        gas_pressure_list.append(gas_pressure_value)

        # total pressures
        total_pressure_values = gas_pressure_value + radiation_pressure_value
        total_pressure_list2.append(total_pressure_values)

        # electron pressure vs gas pressure ratio
        beta_values = electron_pressure_values / gas_pressure_value
        beta_list.append(beta_values)

        # solving for the nuclear particle densities
        n_values = datafunctions.solve_for_nuclear_particle_density(
            nuclear_pressure_values, k, temperature_layers[i]
        )
        n_list.append(n_values)

        # solving for number densities
        n_H2 = datafunctions.solve_for_n_k(abundance_H, n_values)
        n_He2 = datafunctions.solve_for_n_k(abundance_He, n_values)
        n_Ca2 = datafunctions.solve_for_n_k(abundance_Ca, n_values)

        # Saha equations and ionization fractions
        Y11 = (electron_number_density_list[i] ** -1) * datafunctions.general_phi(
            1,
            1,
            temperature_layers[i],
            temperature_partition_function_values,
            C_phi,
            upot_values,
            chi_partition_function_values,
            k,
        )
        f11val = datafunctions.f11(Y11)
        Y12 = (electron_number_density_list[i] ** -1) * datafunctions.general_phi(
            1,
            2,
            temperature_layers[i],
            temperature_partition_function_values,
            C_phi,
            upot_values,
            chi_partition_function_values,
            k,
        )
        Y22 = (electron_number_density_list[i] ** -1) * datafunctions.general_phi(
            2,
            2,
            temperature_layers[i],
            temperature_partition_function_values,
            C_phi,
            upot_values,
            chi_partition_function_values,
            k,
        )
        f12_val = datafunctions.f12(Y12, Y22)
        f22_val = datafunctions.f22(f12_val, Y12)
        Y13 = (electron_number_density_list[i] ** -1) * datafunctions.general_phi(
            1,
            3,
            temperature_layers[i],
            temperature_partition_function_values,
            C_phi,
            upot_values,
            chi_partition_function_values,
            k,
        )
        Y23 = (electron_number_density_list[i] ** -1) * datafunctions.general_phi(
            2,
            3,
            temperature_layers[i],
            temperature_partition_function_values,
            C_phi,
            upot_values,
            chi_partition_function_values,
            k,
        )
        f13_val = datafunctions.f13(Y13, Y23)
        f23_val = datafunctions.f23(f13_val, Y13)
        f33_val = datafunctions.f33(f23_val, Y23)
        f32_value = datafunctions.f32(f22_val, Y22)
        f21_val = datafunctions.f21(f11val, Y11)

        # Ionization fractions for hydrogen
        f11_hydrogen_list.append(f11val)
        f21_hydrogen_list.append(f21_val)
        n_H_list.append(n_H2)

        # Ionization fractions for helium
        f12_helium_list.append(f12_val)
        f22_helium_list.append(f22_val)
        f32_helium_list.append(f32_value)

        # Ionization fractions for calcium
        f13_calcium_list.append(f13_val)
        f23_calcium_list.append(f23_val)
        f33_calcium_list.append(f33_val)

        # atom/ion density for hydrogen
        n11_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f11val, n_H2
        )  # HI density
        n21_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f21_val, n_H2
        )  # HII density

        n11_hydrogen_list.append(n11_v)
        n21_hydrogen_list.append(n21_v)

        # atom/ion density for helium
        n12_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f12_val, n_He2
        )  # HeI density
        n22_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f22_val, n_He2
        )  # HeII density
        n32_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f32_value, n_He2
        )  # HeIII density

        n12_helium_list.append(n12_v)
        n22_helium_list.append(n22_v)
        n32_helium_list.append(n32_v)

        # atom/ion density for calcium
        n13_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f13_val, n_Ca2
        )  # HeI density
        n23_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f23_val, n_Ca2
        )  # HeII density
        n33_v = datafunctions.solve_for_number_densities_of_ionization_stages(
            f33_val, n_Ca2
        )  # HeIII density

        n13_calcium_list.append(n13_v)
        n23_calcium_list.append(n23_v)
        n33_calcium_list.append(n33_v)

        # optical depths
        phys_vs_opt_values = datafunctions.physical_vs_optical_depth(
            log_tau_values[i + 1],
            log_tau_values[i],
            opacity_layers_list[i],
            rho_layers_list[i],
            phys_vs_opt_list[i],
        )
        phys_vs_opt_list.append(phys_vs_opt_values)

        # Hydrogen excitation density fractions
        # n=1
        Hydrogen_excitation_values_n1 = (
            datafunctions.hydrogen_excitation_functions(
                1, temperature_layers[i], R_1, k, interpolated_H_partition_function
            )
            * f11val
        )
        Hydrogen_excitation_values_n1_list.append(Hydrogen_excitation_values_n1)

        # n=2
        Hydrogen_excitation_values_n2 = (
            datafunctions.hydrogen_excitation_functions(
                2, temperature_layers[i], R_1, k, interpolated_H_partition_function
            )
            * f11val
        )
        Hydrogen_excitation_values_n2_list.append(Hydrogen_excitation_values_n2)

        # n=3
        Hydrogen_excitation_values_n3 = (
            datafunctions.hydrogen_excitation_functions(
                3, temperature_layers[i], R_1, k, interpolated_H_partition_function
            )
            * f11val
        )
        Hydrogen_excitation_values_n3_list.append(Hydrogen_excitation_values_n3)
except IndexError as e:
    print(f"{e}")


# adding the final values onto the Hydrogen_excitation_values lists
# because they were not added in the loop above for some reason

Y11_n = (electron_number_density_list[9] ** -1) * datafunctions.general_phi(
    1,
    1,
    temperature_layers[9],
    temperature_partition_function_values,
    C_phi,
    upot_values,
    chi_partition_function_values,
    k,
)
f11val_n = datafunctions.f11(Y11_n)
fin_val = (
    datafunctions.hydrogen_excitation_functions(
        1, temperature_layers[9], R_1, k, interpolated_H_partition_function
    )
    * f11val_n
)
Hydrogen_excitation_values_n1_list.append(fin_val)
Hydrogen_excitation_values_n2_list.append(fin_val)
Hydrogen_excitation_values_n3_list.append(fin_val)


###################################################################################
################################ Schwarzschild Criterion ##########################
###################################################################################

try:
    for i in tqdm(
        range(0, len(pressure_layers_list)), desc="Computing Schwarzschild Criterion"
    ):
        r_rd, ad = datafunctions.Schwarzchild_Criterion(
            i + 1,
            i,
            gas_pressure_list,
            pressure_layers_list,
            opacity_layers_list,
            surface_gravity_g,
            T_eff,
            temperature_layers,
            interpolated_tau_values,
            log_tau_values,
        )

        if r_rd > ad:
            print("Layer " + str(i + 1) + " is unstable to convection")
        else:
            print("Layer " + str(i + 1) + " is stable against convection")
        print(r_rd, ad)
except IndexError as e:
    print(f"{e}")


###################################################################
################################ Figures ##########################
###################################################################


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 6))

ax1.plot(log_tau_values, phys_vs_opt_list)
ax1.set_xscale("log")
ax1.set_title("Depth Comparisions", fontweight="bold", fontsize="13")
ax1.set_xlabel(r"Optical Depth ($\tau$)", fontweight="bold", fontsize="12")
ax1.set_ylabel("Physical Depth ($x$)", fontweight="bold", fontsize="12")

ax2.plot(log_tau_values, temperature_layers_log)
ax2.set_xscale("log")
ax2.set_title("Temperatures Per Layer", fontweight="bold", fontsize="13")
ax2.set_xlabel(r"Optical Depth ($\tau$)", fontweight="bold", fontsize="12")
ax2.set_ylabel("Temperatures ($K$)", fontweight="bold", fontsize="12")

ax3.plot(log_tau_values, opacity_layers_list)
ax3.set_xscale("log")
ax3.set_yscale("log")

ax3.set_title("Opacities Per Layer", fontweight="bold", fontsize="13")
ax3.set_xlabel(r"Optical Depth ($\tau$)", fontweight="bold", fontsize="12")
ax3.set_ylabel("Opacity ($cm^2/g$)", fontweight="bold", fontsize="12")

ax4.plot(log_tau_values, np.log10(rho_layers_list))
ax4.set_xscale("log")
ax4.set_title("Total Mass Density Per Layer", fontweight="bold", fontsize="13")
ax4.set_xlabel(r"Optical Depth ($\tau$)", fontweight="bold", fontsize="12")
ax4.set_ylabel("Total Mass Density ($g/cm^3$)", fontweight="bold", fontsize="12")

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(6, 6))

plt.plot(log_tau_values, beta_list, label=(r"$P_e/P_g$"))
plt.xscale("log")
plt.xlabel(r"Optical Depth ($\tau$)", fontweight="bold", fontsize="12")
plt.title("Pressure Ratio", fontweight="bold", fontsize="14")
plt.ylabel("$P_e/P_g$", fontweight="bold", fontsize="12")
plt.legend(fancybox=True, frameon=True, fontsize=14, shadow=True)
plt.tight_layout()
fig.savefig(path_to_save_figures + "beta.png")
plt.show()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(
    log_tau_values,
    np.log10(radiation_pressure_list),
    label="Radiation Pressure ($dynes/cm^{-2}$)",
)
ax1.plot(
    log_tau_values,
    np.log10(electron_pressures_list),
    label="Electron Pressure ($dynes/cm^{-2}$)",
)
ax1.plot(
    log_tau_values,
    np.log10(total_pressure_list2),
    label="Total Pressure ($dynes/cm^{-2}$)",
)
ax1.plot(
    log_tau_values, np.log10(gas_pressure_list), label="Gas Pressure ($dynes/cm^{-2}$)"
)
ax1.plot(
    log_tau_values,
    np.log10(nuclear_pressures_list),
    label=" Nuclear Pressure ($dynes/cm^{-2}$)",
)
ax1.set_xscale("log")
ax1.set_xlabel("tau", fontweight="bold", fontsize="12")
ax1.set_title("Pressure Comparisions", fontweight="bold", fontsize="14")
ax1.set_ylabel("Pressures ($dynes/cm^{-2}$)", fontweight="bold", fontsize="12")

ax2.plot(log_tau_values, beta_list, label=(r"$P_e/P_g$"))
ax2.set_xscale("log")
ax2.set_xlabel("tau", fontweight="bold", fontsize="12")
ax2.set_title("Pressure Ratio", fontweight="bold", fontsize="14")
ax2.set_ylabel("$P_e/P_g$", fontweight="bold", fontsize="12")

ax1.legend(fancybox=True, frameon=True, framealpha=1, shadow=True)
ax2.legend(fancybox=True, frameon=True, fontsize=14, shadow=True)
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(12, 6))
plt.plot(log_tau_values, f11_hydrogen_list, label="$H^0$")
plt.plot(log_tau_values, f21_hydrogen_list, label="$H^+$")

plt.plot(log_tau_values, f12_helium_list, label="$He^0$")
plt.plot(log_tau_values, f22_helium_list, label="$He^+$")
plt.plot(log_tau_values, f32_helium_list, label="$He^{++}$")

plt.plot(log_tau_values, f13_calcium_list, label="$Ca^0$")
plt.plot(log_tau_values, f23_calcium_list, label="$Ca^+$")
plt.plot(log_tau_values, f33_calcium_list, label="$Ca^{++}$")
plt.xscale("log")
plt.title("All Species Ionization Fractions", fontweight="bold", size=14)
plt.xlabel(r"Optical Depth ($\tau$)", size=12, fontweight="bold")
plt.ylabel("Ionization Fractions", fontweight="bold", size=12)
plt.xscale("log")
plt.tight_layout()

plt.legend(
    fancybox=True,
    frameon=True,
    framealpha=1,
    ncol=1,
    bbox_to_anchor=(1.05, 1),
    loc=2,
    borderaxespad=0.0,
    shadow=True,
)
fig.savefig(path_to_save_figures + "ionizationfractions.png")
plt.show()


plt.figure(figsize=(12, 6))
plt.title("Total Electron Density", size=14, fontweight="bold")
plt.plot(log_tau_values, np.log10(n11_hydrogen_list), label="$H^0$ Number Density")
plt.plot(log_tau_values, np.log10(n21_hydrogen_list), label="$H^+$ Number Density")
plt.plot(log_tau_values, np.log10(n12_helium_list), label="$He^0$ Number Density")
plt.plot(log_tau_values, np.log10(n22_helium_list), label="$He^+$ Number Density")
plt.plot(log_tau_values, np.log10(n32_helium_list), label="$He^{++}$ Number Density")
plt.plot(log_tau_values, np.log10(n13_calcium_list), label="$Ca^0$ Number Density")
plt.plot(log_tau_values, np.log10(n23_calcium_list), label="$Ca^+$ Number Density")
plt.plot(log_tau_values, np.log10(n33_calcium_list), label="$Ca^{++}$ Number Density")
plt.xlabel(r"Optical Depth ($\tau$)", size=12, fontweight="bold")
plt.ylabel("Free Electrons [$cm^{-3}$]", fontweight="bold", size=12)
plt.xscale("log")
plt.tight_layout()
plt.legend(
    fancybox=True,
    frameon=True,
    framealpha=1,
    ncol=1,
    bbox_to_anchor=(1.05, 1),
    loc=2,
    borderaxespad=0.0,
    shadow=True,
)
plt.show()


fig = plt.figure(figsize=(12, 6))
plt.plot(log_tau_values, (Hydrogen_excitation_values_n1_list), label="$n_{111}/n_1}$")
plt.plot(log_tau_values, (Hydrogen_excitation_values_n2_list), label="$n_{211}/n_1}$")
plt.plot(log_tau_values, (Hydrogen_excitation_values_n3_list), label="$n_{311}/n_1}$")
plt.xscale("log")
plt.yscale("log")
plt.legend(fancybox=True, frameon=True, framealpha=1, shadow=True)
plt.xlabel(r"Optical Depth ($\tau$)", fontweight="bold", size=12)
plt.ylabel("Hydrogen Excitation Density Fractions", fontweight="bold", size=12)
plt.title(
    "Hydrogen Excitation Density Fractions for n=1, n=2, and n=3",
    fontweight="bold",
    size=14,
)
fig.savefig(path_to_save_figures + "hydrogenexcitation.png")
plt.show()
