############################################################################################
################################ Import Necessary Libraries ################################
############################################################################################

import numpy as np
import scipy

#########################################################################################
################################ Interpolation Functions ################################
#########################################################################################


def interpolate_opacities(
    opacity_grid, opacity_values, temperature_grid, save_table=False
):
    """
    Interpolate opacities from an OPAL Table.

    Arguments:
        opacity_grid -- The desired opacity grid.
        opacity_values -- Opacity values to interpolate.
        temperature_grid -- Temperature grid that the opacity values will be interpolated against.

    Keyword Arguments:
        save_table -- Save the newly interpolated table as a txt file (default: {False}).

    Returns:
        Array of interpolated opacities.
    """

    row = []

    for i in range(0, len(temperature_grid)):
        vals = np.interp(opacity_grid, opacity_values, temperature_grid[i])
        row.append(vals)
    row = np.array(row)

    if save_table:
        np.savetxt("interpolatedRs.txt", row, fmt="%10.2f")

    return row


def pressure_surface_gravity_scale(P_0, g_f, g_0):
    """
    Pressure - surface gravity scaling function. Used in conjunction with a plot of P_g vs S_0
    from Gray in notes.

    Arguments:
        P_0 -- Reference pressure [dynes/cm^2].
        g_f -- Surface gravity of new star [log].
        g_0 -- Reference surface gravity [log].

    Returns:
        New pressure scaled to match surface gravity of new star [dynes/cm^2].
    """

    P_tau = P_0 + (2 / 3) * (g_f - g_0)

    return P_tau


def hopf_function(Teff, tau, q_tau):
    """
    Calculate temperature as a function of optical depth for each layer in the atmosphere

    Arguments:
        Teff -- Effective temperature of choosen star [K].
        tau -- Optical depth for particular layer.
        q_tau -- Hopf function, determined based on each optical depth value.

    Returns:
        The temperature at each layer as a function of optical depth.
    """

    T_tau = (Teff) * ((3 / 4) * (tau + q_tau)) ** (1 / 4)
    return T_tau


def logRvsT(rho, T):
    """
    Obtains the logged density value.

    Arguments:
        rho -- Density; dependent on pressure [g/cm^3].
        T -- Temperature [K].

    Returns:
        Logged density value.
    """

    T6 = T / (10**6)
    R = rho / (T6**3)
    return np.log10(R)


def column_number(logR_values, opacity_grid):
    """
    Finds the column number corresponding to the closest match of log(R) from
    logRvsT definition and opacity_grid list.
    Reconfigured the interpolation to have more values to get different opacities.

    Arguments:
        logR_values -- logged density value calculated from logRvsT def.
        opacity_grid -- Desired opacity grid

    Returns:
        Returns index for column.
    """

    minlist = []

    for i in range(0, len(opacity_grid)):
        pick = opacity_grid[i]
        difference = np.abs(pick - logR_values)
        minlist.append(difference)
        minvalue = minlist.index(min(minlist))

    return minvalue


def obtain_chi_value(temp, logT, row_interp, logT_value, rho_at_layer, opacity_grid):
    """
    Extract the necessary opacity from the OPAL table.

    Arguments:
        temp -- Temperature [K].
        logT -- Temperature [log].
        row_interp -- Interpolated opacity values.
        logT_value -- np.log10(temperature) value at current layer.
        rho_at_layer -- Density at current layer [g/cm^3].
        opacity_grid -- Desired opacity grid.

    Returns:
        The matching opacity from the OPAL table.

    """

    # calcalates the log(R) value using the above definition and temp
    # temp is from temperature_layers array
    logR_result = logRvsT(rho_at_layer, temp)
    # finds the column number corresponding to the closest value of log(R) found from
    # logR_result that is in the R_values array
    minvalue = column_number(logR_result, opacity_grid)
    # uses the minvalue to interpolate the temperatures from the OPAL table for that
    # particular column corresponding to the closest R value found
    temperature_interpolated = scipy.interpolate.interp1d(
        logT, row_interp[minvalue], kind="cubic"
    )
    # use that interpolated temperature list and input the needed temperature to
    # obtain the correct chi_value
    result = temperature_interpolated(logT_value)
    return result


def pressure_guess(P_previous, logg, tau_current, tau_previous, chi_value_previous):
    """
    Initial pressure guess for the pressures of all of the layers.  This is used
    in conjunction with the convergence equation.

    Arguments:
        P_previous -- Previous pressure guess at current layer [dynes/cm^2].
        logg -- Surface gravity [log].
        tau_current -- Optical depth of current layer.
        tau_previous -- Optical depth of previous layer.
        chi_value_previous --  Opacity value of previous layer.

    Returns:
        New pressure for current layer [dynes/cm^2].
    """

    P_prime = P_previous + logg * ((tau_current - tau_previous) / (chi_value_previous))
    return P_prime


def next_pressure_guess(
    P_previous, logg, tau_current, tau_previous, chi_previous, chi_current
):
    """
    Next pressure guess for current layer. This is used in conjunction with the convergence
    equation.

    Arguments:
        P_previous -- Previous pressure guess at current layer [dynes/cm^2].
        logg -- Surface gravity  [log].
        tau_current -- Optical depth of current layer.
        tau_previous -- Optical depth of previous layer.
        chi_previous --  Opacity value of previous layer.
        chi_current --  Opacity value of current layer .

    Returns:
        Pressure guess [dynes/cm^2].
    """

    P_next = P_previous + 2 * logg * (
        (tau_current - tau_previous) / (chi_previous + chi_current)
    )
    return P_next


def physical_vs_optical_depth(tau_val2, tau_val1, chi, rho, x_i):
    """
    Calculate the physical depth  vs optical depth of the layers.

    Arguments:
        tau_val2 -- Optical depth of next layer.
        tau_val1 -- Optical depth of current layer.
        chi -- Opacity of current layer.
        rho -- Density of current layer [g/cm^-3].
        x_i -- Optical depth of layers. x_0 = 0 for the surface.

    Returns:
        The optical depth of the layer.
    """

    x_opt = ((tau_val2 - tau_val1) / (chi * rho)) + x_i
    return x_opt


# All the pressure related definitions are here.
def solve_for_nuclear_particle_density(P, k, T):
    """
    Solve for the nuclear particle density.

    Arguments:
        P -- Nuclear pressure.
        k -- Boltzmann's constant [cm^2 g s^-2 K^-1].
        T -- Temperature [K].

    Returns:
        Nuclear particle density.
    """

    n = (P) / (k * T)
    return n


def solve_for_n_k(alpha_k, n_N):
    """
    Obtain the number density of a specific metal. Use with
    the abundance_fraction definition.

    Arguments:
        alpha_k -- Abundance fraction of a specific metal.
        n_N -- Nuclear density of all atomic particles.

    Returns:
        Number density of a specific metal.
    """

    n_k = alpha_k * n_N
    return n_k


def electron_pressure(k, T, ne):
    """
    Calculate the electron pressure.

    Arguments:
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].
        T -- temperature [K].
        ne -- Electron density (obtained from the trascendential equation).

    Returns:
        Electron pressure.
    """

    e_pressure = (ne) * (k) * (T)
    return e_pressure


def nuclear_pressure(k, T, nN):
    """
    Calculate the nuclear pressure.

    Arguments:
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].
        T -- temperature [K].
        nN -- Nuclear density.

    Returns:
        Nuclear pressure.
    """

    n_pressure = (nN) * (k) * (T)
    return n_pressure


def nuclear_pressure2(rho, T, mu_N, k, ma):
    """
    Calculate the nuclear pressure.

    Arguments:
        rho -- Total density per layer [g/cm^3].
        T -- Temperature [K].
        mu_N -- >ean molecular weight of all nuclear particles.
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].
        ma -- Atomic weight mass to g.

    Returns:
        Nuclear pressure.
    """

    n_pressure2 = (k * rho * T) / (mu_N * ma)
    return n_pressure2


def radiation_pressure(T, a):
    """
    Calculate the radiation pressure.

    Arguments:
        T -- temperature [K].
        a -- Constant [erg cm^-3 K^-4].

    Returns:
        Radiation pressure.
    """

    radiation_pressure = (a / 3) * (T**4)
    return radiation_pressure


def nuclear_n_from_obtain_rho(P, T, ne, k):
    """
    Obtain the total density.

    Arguments:
        P -- Pressure at layer [dynes/cm^2].
        T -- Temperature at layer [K].
        ne --  Electron density obtained from transcendential equation.
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].

    Returns:
        Total density [g/cm^3].
    """

    n_nuclear = solve_for_nuclear_particle_density(P, k, T)
    n_total = n_nuclear - ne
    return n_total


def transcential_eq(
    ne,
    P,
    T,
    k,
    C_phi,
    upot_values,
    chi_values,
    abundances_H,
    abundances_He,
    abundances_Ca,
    temperature_list,
):
    """
    Computes the transcential equation to solve for ne. Use in tandem with the
    scipy.optimize.brentq module.

    Arguments:
        ne -- Electron number density.
        P -- Pressure [dynes/cm^2].
        T -- Temperature [K].
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].
        C_phi -- Partition function constant.
        upot_values -- Partition function.
        chi_values -- Opacity partitition function values.
        abundances_H -- Abundance fractions of Hydrogen.
        abundances_He -- Abundance fractions of Helium.
        abundances_Ca -- Abundance fractions of Calcium.
        temperature_list -- Temperatures [K].

    Returns:
        Used in conjunction with the brentq method to solve for ne.
    """

    Y11 = (ne**-1) * general_phi(
        1, 1, T, temperature_list, C_phi, upot_values, chi_values, k
    )
    f11_val = f11(Y11)
    Y12 = (ne**-1) * general_phi(
        1, 2, T, temperature_list, C_phi, upot_values, chi_values, k
    )
    Y22 = (ne**-1) * general_phi(
        2, 2, T, temperature_list, C_phi, upot_values, chi_values, k
    )
    f12_val = f12(Y12, Y22)
    f22_val = f22(f12_val, Y12)
    Y13 = (ne**-1) * general_phi(
        1, 3, T, temperature_list, C_phi, upot_values, chi_values, k
    )
    Y23 = (ne**-1) * general_phi(
        2, 3, T, temperature_list, C_phi, upot_values, chi_values, k
    )
    f13_val = f13(Y13, Y23)
    f23_val = f23(f13_val, Y13)

    nt = solve_for_nuclear_particle_density(P, k, T)
    first_term = nt - ne

    abundances_H = abundances_H * ((0) * (f11(Y11)) + (1) * (f21(f11_val, Y11)))
    abundances_He = abundances_He * (
        (0) * (f12(Y12, Y22)) + (1) * (f22(f12_val, Y12)) + (2) * (f32(f22_val, Y22))
    )
    abundances_Ca = abundances_Ca * (
        (0) * (f13(Y13, Y23)) + (1) * (f23(f13_val, Y13)) + (2) * (f33(f23_val, Y23))
    )

    second_term = abundances_H + abundances_He + abundances_Ca
    third_term = first_term * second_term
    fourth_term = ne
    final = third_term - fourth_term
    return final


# Definitions related to abundance fractions and number density of metals.
def mu_n(x_1, A_1, x_2, A_2, x_3, A_3):
    """
    Obtain the mean molecular weight for nuclear particles.

    Arguments:
        x_1 -- Mass fraction of H.
        A_1 -- Atomic weight of H.
        x_2 -- Mass fraction of He.
        A_2 -- Atomic weight of He.
        x_3 -- Mass fraction of metal.
        A_3 -- Mass fraction of metal.

    Returns:
        Mean molecule weight for nuclear particles.
    """

    alpha_1 = x_1 / A_1
    alpha_2 = x_2 / A_2
    alpha_3 = x_3 / A_3
    mu = alpha_1 + alpha_2 + alpha_3
    mu_n = (mu) ** (-1)
    return mu_n


def abundance_fraction(x_1, A_1, x_2, A_2, x_3, A_3):
    """
    Obtain the abundance ratio for a specific metal (the first inputted metal).

    Arguments:
        x_1 -- Mass fraction of H.
        A_1 -- Atomic weight of H.
        x_2 -- Mass fraction of He.
        A_2 -- Atomic weight of He.
        x_3 -- Mass fraction of metal.
        A_3 -- Mass fraction of metal.

    Returns:
        Abundance ratio for the first inputted metal.
    """

    alpha_1 = x_1 / A_1
    sum_alpha = alpha_1 + (x_2 / A_2) + (x_3 / A_3)
    abundance_1 = alpha_1 / sum_alpha
    return abundance_1


def solve_for_number_densities_of_ionization_stages(f_jk, n_k):
    """
    Solve the number density of a particular species/metal.

    Arguments:
        f_jk -- Inionization fraction for a specific species/metal.
        n_k -- Number density for that particular species/metal.

    Returns:
        Number density of a particular species/metal.
    """

    n_jk = f_jk * n_k
    return n_jk


# Definitions related to the ionization fractions of species, the Saha Equation, and the] partitiion function.
def f13(Y13, Y23):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f13 = 1 + Y13 + Y13 * Y23
    return 1 / f13


def f23(f13, Y13):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f23 = f13 * Y13
    return f23


def f33(f23, Y23):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f33 = f23 * Y23
    return f33


def f11(Y11):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f11 = 1 + Y11
    return 1 / f11


def f21(f11, Y11):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f21 = f11 * Y11
    return f21


def f12(Y12, Y22):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f12 = 1 + Y12 + Y22 * Y12
    return 1 / f12


def f22(f12, Y12):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f22 = f12 * Y12
    return f22


def f32(f22, Y22):
    """Uses the calculated results from the Saha Equations later on in the code."""

    f32 = f22 * Y22
    return f32


def general_phi(j, kg, T, temperature_list, C_phi, upot_table, chi_table, k):
    """
    The generalized Phi(T) value for a given ionization stage (j), species
    (kg), and temperature (T).

    Arguments:
        j -- Given ionization stage.
        kg -- Given species.
        T -- Temperature [K].
        temperature_list -- Temperaures [K].
        C_phi -- Partition function constant.
        upot_values -- Partition function.
        chi_values -- Opacity partitition function values.
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].

    Returns:
        The general Phi(T) value.
    """

    # gives you index for column for which theta matches
    minlist = []
    for i in range(0, len(temperature_list)):
        pick = temperature_list[i]
        difference = np.abs(pick - T)
        minlist.append(difference)
        minvalue = minlist.index(min(minlist))

    first = C_phi * T ** (3 / 2)

    if kg == 1:
        upot_val1 = upot_table[minvalue][j]
        upot_val2 = upot_table[minvalue][j + 1]
        potential_value = (10**upot_val2) / (10**upot_val1)
        chival = chi_table[j - 1]
        finalval = first * potential_value * np.exp(-chival / (k * T))

    elif kg == 2:
        upot_val1 = upot_table[minvalue][j + 2]
        upot_val2 = upot_table[minvalue][j + 3]
        potential_value = (10**upot_val2) / (10**upot_val1)
        chival = chi_table[j]
        finalval = first * potential_value * np.exp(-chival / (k * T))

    elif kg == 3:
        upot_val1 = upot_table[minvalue][j + 5]
        upot_val2 = upot_table[minvalue][j + 6]
        potential_value = (10**upot_val2) / (10**upot_val1)
        chival = chi_table[j + 2]
        finalval = first * potential_value * np.exp(-chival / (k * T))

    return finalval


def U_11_value(T, interpolated_partition_function):
    """
    Obtain the U_11 value for an temperature.

    Arguments:
        T -- Temperature [K].
        interpolated_partition_function -- Interpolated partition function.

    Returns:
        Returns a numerical value for U_11(T).
    """

    U11_final = interpolated_partition_function(5040 / T)
    return U11_final * 1


def hydrogen_excitation_functions(n, T, R_1, k, interpolated_partition_function):
    """
    Calculate the Hydrogen excitation values.

    Arguments:
        n -- Hydrogren level.
        T -- Temperature [K].
        R_1 -- Constant [ergs].
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].
        interpolated_partition_function -- Inteprolated partition function.

    Returns:
        Calculated Hydrogen excitation values.
    """

    g_n11 = 2 * (n**2)
    chi_n11 = R_1 * (1 - (1 / (n**2)))
    # U_11 = 10**(general_phi2(T))
    U_11 = 10 ** (U_11_value(T, interpolated_partition_function))
    first_term = g_n11 / U_11
    second_term = np.exp(-(chi_n11 / (k * T)))
    full_term = first_term * second_term
    return full_term


def rho_total(n_1, A_1, n_2, A_2, n_3, A_3, ne, ma, me):
    """
    Obtain total density per layer. Use with the obtain_rho
    definition.

    Arguments:
        n_1 -- Density of H.
        A_1 -- Atomic weight of H.
        n_2 -- Density of He.
        A_2 -- Atomic weight of He.
        n_3 -- Density of metal.
        A_3 -- Atomic weight of metal.
        ne -- Electron density to grams.
        ma -- Atomic weight mass.
        me -- Mass of an electron [g].

    Returns:
        Total density per layer [g/cm^3].
    """

    sum1 = n_1 * A_1 * ma
    sum2 = n_2 * A_2 * ma
    sum3 = n_3 * A_3 * ma
    electron_contribution = me * ne
    rho_total = sum1 + sum2 + sum3 + electron_contribution
    return rho_total


def obtain_rho(ne, P, T, k, X, A_H, Y, A_He, Z, A_Ca, ma, me):
    """
    Obtain total density per layer. Use with rho_total definition.

    Arguments:
        ne -- Electron density obtained from the transcendential eq.
        P -- Pressure per layer [dynes/cm^2].
        T -- Temperature per layer [K].
        k -- Boltzmann's Constant [cm^2 g s^-2 K^-1].
        X -- Mass fraction of Hydrogen (H).
        A_H -- Atomic weight of Hydrogen.
        Y -- Mass fraction of Helium (He).
        A_He -- Atomic weight of Helium.
        Z -- Mass fraction of Calcium.
        A_Ca -- Atomic weight of Calcium.
        ma -- Atomic weight mass.
        me -- Mass of an electron [g].

    Returns:
        Total density per layer [g/cm^3].
    """

    n_nuclear = solve_for_nuclear_particle_density(P, k, T)
    abundance_H = abundance_fraction(X, A_H, Y, A_He, Z, A_Ca)
    abundance_He = abundance_fraction(Y, A_He, X, A_H, Z, A_Ca)
    abundance_Ca = abundance_fraction(Z, A_Ca, X, A_H, Y, A_He)
    n_H = solve_for_n_k(abundance_H, n_nuclear)
    n_He = solve_for_n_k(abundance_He, n_nuclear)
    n_Ca = solve_for_n_k(abundance_Ca, n_nuclear)
    rho_tots = rho_total(n_H, X, n_He, Y, n_Ca, Z, ne, ma, me)
    return rho_tots


def convergence(
    P0,
    initial_num,
    final_num,
    logg,
    tau,
    opacity_layers_list,
    temperature_layers,
    temperature_layers_log,
    rho_layers_list,
    pressure_layers_list,
    opal_logT,
    rho_interp,
    opacity_grid,
):
    """
    Convergence Code. This convergences the pressure within a certain
    tolerance limit designated by epsilon so that the upward pressure and the
    downwar pressure match.

    Arguments:
        P0 -- Pressure corresponding to each layer [dynes/cm^2].
        initial_num -- Current layer.
        final_num -- Next layer.
        logg -- Surface gravity [log].
        tau -- Pptical depth.
        opacity_layers_list -- Opacity values.
        temperature_layers -- Temperatures [K].
        temperature_layers_log -- temperatures [log].
        rho_layers_list -- Densities [g/cm^3].
        pressure_layers_list -- Pressures [dynes/cm^2].
        opal_logT -- OPAL temperatures [log].
        rho_interp -- Interpolated temperatures.
        opacity_grid -- Desired opacity grid.

    Returns:
        The converged pressure [dynes/cm^2].
    """

    loops = 0
    # tolerance level in the if statement
    epsilon = 0.0001  # 10**(-5)

    # first pressure guess
    p_guess = pressure_guess(
        P0, logg, tau[final_num], tau[initial_num], opacity_layers_list[initial_num]
    )
    next_chi = obtain_chi_value(
        temperature_layers[final_num],
        opal_logT,
        rho_interp,
        temperature_layers_log[final_num],
        rho_layers_list[initial_num],
        opacity_grid,
    )
    next_chi = 10 ** (next_chi)
    # adds the final opacity to a list
    # modified pressure guess
    next_p = next_pressure_guess(
        p_guess,
        logg,
        tau[final_num],
        tau[initial_num],
        opacity_layers_list[initial_num],
        next_chi,
    )

    # convergence loop
    while True:
        loops = loops + 1
        if np.abs(next_p - p_guess) / p_guess <= epsilon:
            P0 = next_p
            # adds correct pressure to list
            pressure_layers_list.append(P0)
            final_chi = next_chi
            #             print('This is it: ' + str(np.log10(P0)))
            #             print(opacity_layers_list[initial_num], final_chi)
            opacity_layers_list.append(final_chi)
            break

        else:
            p_guess = next_p
            next_chi = obtain_chi_value(
                temperature_layers[final_num],
                opal_logT,
                rho_interp,
                temperature_layers_log[final_num],
                rho_layers_list[initial_num],
                opacity_grid,
            )
            next_chi = 10 ** (next_chi)
            next_p = next_pressure_guess(
                p_guess,
                logg,
                tau[final_num],
                tau[initial_num],
                opacity_layers_list[initial_num],
                next_chi,
            )
    return P0


# Definitions related to the Schwarzschild Criterion that establishes whether a layer is stable (ad < r*r_d) or unstable (ad > r*r_d) to convection. '''
def r_d(fin_num, initial_num, tau_values_interpolation, tau):
    """
    Obtain the change in r_d for the Schwarzschild Criterion.

    Arguments:
        fin_num -- i+1 (next layer).
        initial_num -- i (current layer).
        tau_values_interpolation -- Interpolated optical depth.
        tau -- Optical depth for each layer..

    Returns:
        change in r_d for the Schwarzschild Criterion.
    """

    value = 1 + (
        (
            1 * tau_values_interpolation(tau[fin_num])
            - 1 * tau_values_interpolation(tau[initial_num])
        )
        / (tau[fin_num] - tau[initial_num])
    )
    return value


def changer(
    initial_num,
    opacity_layers,
    pressure_layers,
    surface_grav,
    effective_temp,
    temperature_layers,
):
    """
    Obtain r for the Schwarzschild Criterion.

    Arguments:
        initial_num --  i (current layer)
        opacity_layers -- Opacities for each layer.
        pressure_layers -- Pressures for each layer.[dynes/cm^2].
        surface_grav -- Surface gravity [log]
        effective_temp -- Effective Temperature [K].
        temperature_layers -- Temperature for each layer.[K].

    Returns:
        R for the Schwarzschild Criterion.
    """

    mult1 = 3 / 16
    mult2 = (opacity_layers[initial_num] * pressure_layers[initial_num]) / (
        10**surface_grav
    )
    mult3 = (effective_temp / temperature_layers[initial_num]) ** 4
    val = mult1 * mult2 * mult3
    return val


def change_ad(P_gas, Pressure):
    """
    Obtain the change in adiabatic pressure of layers for the Schwarzschild Criterion.

    Arguments:
        P_gas -- Gas pressure for each layer
        Pressure --  Pressures for each layer [dynes/cm^2].

    Returns:
        The change in the adiabatic pressure for the Schwarzschild Criterion.
    """

    beta = P_gas / Pressure
    first_beta_term = (1 / (beta**2)) * ((1 - beta) * (1 + beta))
    add_first_beta_term = 1 + first_beta_term
    second_beta_term = (4 / (beta**2)) * ((1 - beta) * (1 + beta))
    add_second_beta_term = (5 / 2) + second_beta_term
    total = add_first_beta_term / add_second_beta_term
    return total


def Schwarzchild_Criterion(
    fin_num,
    initial_num,
    gas_pressure_layers,
    pressure_layers,
    opacity_layers,
    surface_grav,
    effective_temp,
    temperature_layers,
    tau_values_interpolation,
    tau,
):
    """
    The Schwarzschild Criterion that tells us if a layer is
    stable (ad < r*r_d) or unstable (ad > r*r_d) to convection.

    Arguments:
        fin_num -- i+1 (the next layer)
        initial_num -- i (the current layer)
        gas_pressure_layers -- Gas Pressures for each layer.
        pressure_layers -- Pressures for each layer.[dynes/cm^2].
        opacity_layers -- Opacities for each layer.
        surface_grav -- Surface gravity [log]
        effective_temp -- Effective Temperature [K].
        temperature_layers -- Temperature for each layer.[K].
        tau_values_interpolation -- Interpolated optical depth.
        tau -- Optical depth for each layer.

    Returns:
        If stable or unstable to convection.
    """

    change_r_rd = changer(
        initial_num,
        opacity_layers,
        pressure_layers,
        surface_grav,
        effective_temp,
        temperature_layers,
    ) * r_d(fin_num, initial_num, tau_values_interpolation, tau)

    change_ad_v = change_ad(
        gas_pressure_layers[initial_num],
        pressure_layers[initial_num],
    )

    return change_r_rd, change_ad_v
