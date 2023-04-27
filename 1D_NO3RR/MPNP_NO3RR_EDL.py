#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: divyabohra (CO2 reduction script)
modified by Paige Brimley and Matthew Liu

This script solves the GMPNP system for NO3RR for steady state \
concentration of solution species as well as value of potential and \
electric field as a function of space and time.

Dirichlet conditions are used at both the left and right boundaries \
for potential and for concentration of species in the bulk.\
Flux boundary conditions are used at the OHP for all species

The geometry and the mesh are generated using a separate script.

4 heterogeneous reactions were considered in the original CO2RR model:
    CO2 + H2O + 2e- -> CO + 2OH-
    2H2O + 2e- -> H2 + 2OH-
    2(H+ + e-) -> H2
    CO2 + 2(H+ + e-) -> CO + H2O

In this adaptation, we consider these 3 heterogeneous reactions:
    NO3- + H2O + 2 e- -> NO2- + 2 OH-

    NO3- + 7H2O + 8e- -> NH4+ + 10 OH-

    2H2O + 2e- -> H2 + 2OH-

The rates of the above reactions are input to the simulation in the form of partial current density data. The
electrode reactions are modeled as flux boundary conditions across the OHP, where species that do not participate in
electrode reactions have a zero flux condition imposed.

3 homogeneous reactions were considered in the original CO2RR model:
    H2O <=> H+ + OH- (k_w1, k_w2)
    HCO3- + OH- <=> CO32- + H2O (k_a1, k_a2)
    CO2 + OH- <=> HCO3- (k_b1, k_b2)

In this adaptation, we consider these homogeneous reactions:
    H2O <=> H+ + OH- (k_w1, k_w2)
    NH4+ + OH- <=> NH3 + H2O
    NO2- + H2O -> HNO2 + OH-

The values of the forward and backward rate constants are taken from literature and can be found in parameters.yaml
file. We use equilibrium constants for homogenous reactions with N-containing species.

Species solved for (i) were previously: H+, OH-, HCO3-, CO32-, CO2, cat+
now we solve for: H+, OH-, NO3-,NO2-, NH4+, NH3, HNO2, CLO4-, and cat+

The electrolyte assumed in the original CO2RR model was 0.1 M KHCO3. In this model, we are assuming 1.0 M NaClO4 + 10
mM HNO3 (although the NaClO4 concentration can easily be adjusted to lower concentrations, e.g. 10 mM)

"""

from __future__ import print_function
from fenics import *  # pylint: disable=unused-wildcard-import
import yaml
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime, date, time
import os
import argparse
import json


# The function below is used to scale back the calculated variables from dimensionless form to SI units tau is scaled
# time, C is scaled concentration
def scale(
        species='H',
        tau=None,
        C=None,
        initial_conc={'H': 0.0},
        L_n=0.0,
        L_debye=0.0):
    diff_ref = 1.4066e-9  # D_NO3, in m2s-1 (L2T-1)
    t = (tau * L_debye * L_n) / diff_ref
    c = C * initial_conc[species]

    return t, c


# main loop function
def solve_EDL(
        concentration_elec=1.0,
        model='MPNP',
        voltage_multiplier=-1.0,  # multiplier to thermal voltage (0.02585 V) and specifies potential vs PZC at OHP.
        H2_FE=0.2,
        NO2_FE=0.3,
        mesh_structure='variable',
        current_OHP_ss=10.0,  # A m-2
        L_n=50.0e-6,
        stabilization='N',
        H_OHP=None,
        cation='Na',
        params_file='parameters',
        dry_run=True
):
    tol = 1.0e-14  # tolerance for coordinate comparisons
    stamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    # change below path for accessing utilities
    basepath_utilities = '/SET BASEPATH HERE/'

    # read rate constants of homogeneous reactions, diffusion coefficients \
    # and diffusion length from yaml file storing default parameters
    f_params = open(basepath_utilities + params_file + '.yaml')
    data = yaml.safe_load(f_params)

    rate_constants = data['rate_constants']

    # see code notes at top for reactions corresponding to rate constants
    # right now, only importing those for water as we are using equilibrium constants for all else
    kw1 = rate_constants['kw1']
    kw2 = rate_constants['kw2']

    # storing the cation string in cat_str
    cat_str = cation

    # hydration numbers of cations
    n_water = {'H': 10.0, cat_str: 0.0}

    if cat_str == 'K':
        n_water[cat_str] = 4
    elif cat_str == 'Li':
        n_water[cat_str] = 5
    elif cat_str == 'Cs':
        n_water[cat_str] = 3
    elif cat_str == 'Na':
        n_water[cat_str] = 5

    # species explicitly solved for in the simulation:
    species = ['H', 'OH', 'NO3', 'NO2', 'NH4', 'NH3', 'ClO4', 'HNO2', cat_str]

    # initializing species diffusion coefficients:
    diff_coeff = {'H': 0.0, 'OH': 0.0, 'NO3': 0.0, 'NO2': 0.0, 'NH4': 0.0, 'NH3': 0.0, 'ClO4': 0.0, 'HNO2': 0.0,
                  cat_str: 0.0}

    # saving diffusion coefficient of solution species (pulling from parameters.yaml):
    for i in species:
        diff_coeff[i] = data['diff_coef']['D_' + i]

    # initializing solvated sizes of solution species:
    solv_size = {'H': 0.0, 'OH': 0.0, 'NO3': 0.0, 'NO2': 0.0, 'NH4': 0.0, 'NH3': 0.0, 'ClO4': 0.0, 'HNO2': 0.0,
                 cat_str: 0.0}

    # saving solvation size data (from parameters.yaml):
    for i in species:
        solv_size[i] = data['solv_size']['a_' + i]

    # Natural constants, all parameter values used are in SI units:
    farad = data['nat_const']['F']  # Faraday's constant
    temp = data['nat_const']['T']  # temperature
    k_B = data['nat_const']['k_B']  # Boltzmann constant
    e_0 = data['nat_const']['e_0']  # elementary electron charge
    eps_0 = data['nat_const']['eps_0']  # permittivity of vacuum
    eps_rel = data['nat_const']['eps_rel']  # relative permittivity of water
    R = data['nat_const']['R']  # gas constant
    N_A = data['nat_const']['N_A']  # Avogadro's number

    f_params.close()

    # Read bulk electrolyte concentrations....
    # For NO3RR, this data comes from experiment since no homogenous reactions occur until after electrolysis:
    f_conc = open(
        basepath_utilities + 'bulk_soln_' + str(concentration_elec) + 'NaClO4_10mmHNO3.yaml'
    )

    data = yaml.safe_load(f_conc)

    # ending pH of the system (doesn't actually doesn't contribute to code at all for NO3RR):
    bulk_pH = data['bulk_conc_post_NO3']['final_pH']

    # Hard coding charge of solution species (must be entered manually below):
    z = {'H': 1, 'OH': -1, 'NO3': -1, 'NO2': -1, 'NH4': 1, 'NH3': 0, 'ClO4': -1, 'HNO2': 0, cat_str: 1}

    # storing bulk concentration of solution species, initialize with 0
    initial_conc = {'H': 0.0, 'OH': 0.0, 'NO3': 0.0, 'NO2': 0.0, 'NH4': 0.0, 'NH3': 0.0, 'ClO4': 0.0, 'HNO2': 0.0,
                    cat_str: 0.0}

    for i in species:
        initial_conc[i] = data['bulk_conc_post_NO3']['concentrations']['C0_' + i]

    f_conc.close()

    # H_OHP is the current density due to proton consumption, assumed to be 0 by default:
    # Note: This is a hold-over from the original CO2 code, H_OHP is always zero for NO3RR simulations
    if H_OHP is None:
        current_H_frac = 0.0
    else:
        current_H_frac = 0.001  # initializing at a low value

    # estimation of the Debye length from a Boltzmann distribution:
    L_debye = sqrt(
        (eps_0 * eps_rel * k_B * temp) /
        (2 * e_0 ** 2 * concentration_elec * 1.0e+3 * N_A)
    )

    L_D = Constant(L_debye / L_n)  # scaled Debye length

    thermal_voltage = (k_B * temp) / e_0  # thermal voltage

    # scaled time constant, using the diffusion coefficient for NO3- as the scaling standard:
    time_constant = L_debye * L_n / diff_coeff['NO3']

    # initialize scaling factor for homogeneous reaction rate stoichiometry
    scale_R = {'H': 0.0, 'OH': 0.0, 'NO3': 0.0, 'NO2': 0.0, 'NH4': 0.0, 'NH3': 0.0, 'ClO4': 0.0, 'HNO2': 0.0,
               cat_str: 0.0}

    # assigning scaling factor to all species:
    for i in species:
        print(str(i))
        scale_R[i] = Constant((L_n ** 2) / (diff_coeff['NO3'] * initial_conc[i]))

    # scaling factors for Poisson equation
    q = Constant((farad ** 2 * L_n ** 2) / (eps_0 * R * temp))

    # scaled volume of solvated ions, initialize with zero:
    scale_vol = {'H': 0.0, 'OH': 0.0, 'NO3': 0.0, 'NO2': 0.0, 'NH4': 0.0, 'NH3': 0.0, 'ClO4': 0.0, 'HNO2': 0.0,
                 cat_str: 0.0}

    for i in species:
        scale_vol[i] = Constant(solv_size[i] ** 3 * initial_conc[i] * N_A)

    # scaling factors for flux boundary conditions, only applies to species that participate in electrode reactions:
    J_H_prefactor = L_n / (diff_coeff['H'] * initial_conc['H'] * farad)
    J_OH_prefactor = L_n / (diff_coeff['OH'] * initial_conc['OH'] * farad)
    J_NO3_prefactor = L_n / (diff_coeff['NO3'] * initial_conc['NO3'] * farad)

    # depending on the time (which determines bulk acidity), we expect that either HNO2 or NO2- will be present in
    # solution. Therefore, we have flux scaling factors for each species but will only end up using one.
    # The default conditions is to assume that NO2- is being formed at the electrode:
    J_NO2_prefactor = L_n / (diff_coeff['NO2'] * initial_conc['NO2'] * farad)
    # J_HNO2_prefactor = L_n / (diff_coeff['HNO2'] * initial_conc['HNO2'] * farad)

    # same conditions as above for NH4+/NH3 equilibrium. We also consider this case due to mechanistic uncertainty:
    J_NH4_prefactor = L_n / (diff_coeff['NH4'] * initial_conc['NH4'] * farad)  # default assumption
    # J_NH3_prefactor = L_n / (diff_coeff['NH3'] * initial_conc['NH3'] * farad)

    # this voltage is at the OHP and not at the electrode surface, it is with respect to the PZC of the surface
    # for NO3RR, this may be a polycrystalline titanium or copper electrode:
    voltage_scaled = Constant(voltage_multiplier)

    # defining folder name to store simulation results
    identifier = 'voltage_' + str(voltage_multiplier) + '_H2_FE_' + str(H2_FE) \
                 + '_current_' + str(current_OHP_ss) \
                 + '_H_OHP_' + str(H_OHP) + '_cation_' + cat_str

    # Assigning the mesh based upon desired, simulated diffusion length
    # Due to the instabilities inherent in solving the GMPNP equations, the meshes are formatted in a variable structure
    # This structure is as follows: 1st 100 nm are spaced in increments of 0.1 nm, any nm after that are spaced in 10 nm
    # Meshes are generated from the gen_mesh_new.py script

    L_sys = int(L_n * 1.0e+6)
    if mesh_structure == 'variable':
        mesh_structure = mesh_structure + '_coarse_fine_' + str(L_sys) + 'um'
        if L_sys == 50:
            mesh_number = 10981
        elif L_sys == 67:
            mesh_number = 7690
        elif L_sys == 70:
            mesh_number = 7990
        elif L_sys == 107:
            mesh_number = 11690
        elif L_sys == 108:
            mesh_number = 11790
        elif L_sys == 217:
            mesh_number = 22690
    elif mesh_structure == 'uniform':
        mesh_number = 1000

    # Read mesh from file
    print(basepath_utilities + '1D' + mesh_structure + '_mesh_'
          + str(mesh_number) + '.xml.gz')
    mesh = Mesh(
        basepath_utilities + '1D_' + mesh_structure + '_mesh_'
        + str(mesh_number) + '.xml.gz'
    )

    # defining boundary where Dirichlet conditions apply
    def boundary_R(x, on_boundary):
        if on_boundary:
            if near(x[0], 1, tol):
                return True
            else:
                return False
        else:
            return False

    # defining boundary where van Neumann conditions apply
    def boundary_L(x, on_boundary):
        if on_boundary:
            if near(x[0], 0, tol):
                return True
            else:
                return False
        else:
            return False

    if dry_run:
        # without staging time for trouble shooting purposes, runs for 1000 time steps (customizable)
        # 1e-5 sufficient for 50 microns, 1e-3 sufficient for 1 micron or less.
        time_step = 1.0e-5
        total_sim_time = 2.0e-2

        T = total_sim_time / time_constant  # final time
        dt = time_step / time_constant  # step size
        num_steps = total_sim_time / time_step  # number of steps
        del_ts = [Constant(dt)]
        del_t = del_ts[0]
        tot_num_steps = int(num_steps)

    else:
        # long time simulations (t > 10s)
        # we use 2 time step sizes serially over the total simulation time to reach steady state
        time_step_1 = 1.0e-6
        time_step_2 = 1.0e-3

        total_sim_time_1 = 0.0001  # in sec
        total_sim_time_2 = 13.1  # in sec

        T_1 = total_sim_time_1 / time_constant  # final time
        dt_1 = time_step_1 / time_constant  # step size
        num_steps_1 = int(total_sim_time_1 / time_step_1)

        T_2 = total_sim_time_2 / time_constant  # final time
        dt_2 = time_step_2 / time_constant  # step size
        num_steps_2 = int((total_sim_time_2 - total_sim_time_1) / time_step_2)

        del_ts = [Constant(dt_1), Constant(dt_2)]
        del_t = del_ts[0]  # start with smaller time step size
        dts = [dt_1, dt_2]
        dt = dts[0]
        tot_num_steps = num_steps_1 + num_steps_2

    # define path to store simulation output folder
    basepath = '1D' + model + '/'

    newpath = basepath + stamp + '_experiment/' + identifier
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Define function space for system of concentrations
    degree = 1
    P1 = FiniteElement('P', interval, degree)
    # Elements = # of species explicitly solved for plus 1 potential
    element = MixedElement([P1, P1, P1, P1, P1, P1, P1, P1, P1, P1])
    V = FunctionSpace(mesh, element)
    W = VectorFunctionSpace(mesh, 'P', degree)
    Y = FunctionSpace(mesh, 'P', degree)

    # Define test functions
    (  # pylint: disable=unbalanced-tuple-unpacking
        v_H,
        v_OH,
        v_NO3,
        v_NO2,
        v_NH4,
        v_NH3,
        v_ClO4,
        v_HNO2,
        v_cat,
        v_p
    ) = TestFunctions(V)

    # Define functions for the concentrations and potential
    u = Function(V)  # at t_n+1
    # initialization of all variables
    u_0 = Expression(
        ('1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '1.0', '0.0'),
        degree=1)
    # initializing concentration as bulk and voltage as grounded
    u_n = project(u_0, V)

    # accessing components at t=t_n+1
    (  # pylint: disable=unbalanced-tuple-unpacking
        u_H,
        u_OH,
        u_NO3,
        u_NO2,
        u_NH4,
        u_NH3,
        u_ClO4,
        u_HNO2,
        u_cat,
        u_p
    ) = split(u)

    # accessing components t=t_n
    (  # pylint: disable=unbalanced-tuple-unpacking
        u_nH,
        u_nOH,
        u_nNO3,
        u_nNO2,
        u_nNH4,
        u_nNH3,
        u_nClO4,
        u_nHNO2,
        u_ncat,
        u_np
    ) = split(u_n)

    bulk = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    # Constant values for concentrations in bulk and grounded potential
    bc1 = DirichletBC(V, bulk, boundary_R)
    # Dirichlet condition for voltage at the OHP
    bc2 = DirichletBC(V.sub(9), voltage_scaled, boundary_L)
    bcs = [bc1, bc2]

    solver_parameters = {
        'nonlinear_solver': 'newton',
        'newton_solver': {
            'maximum_iterations': 100,
            'relative_tolerance': 1.0e-4,
            'absolute_tolerance': 1.0e-4
        }
    }

    ## SETTING THE FLUX AT THE OHP:
    # Note about naming conventions: to keep things simple, I use NH4_FE and NO2_FE for when J_HNO2 and J_NH3 terms are
    # used as inputs

    NH4_FE = 1 - H2_FE - NO2_FE  # FE of NH4+/NH3 production
    # NH4_FE = 0.1619
    # NH4_FE = 0.1619
    J_NO3 = Constant(
        J_NO3_prefactor * current_OHP_ss * 0.5 * (NO2_FE) + J_NO3_prefactor * current_OHP_ss * 0.125 * (NH4_FE)
    )  # 1 / n where n is number of electrons going to NH3 (8)

    J_H = Constant(J_H_prefactor * current_OHP_ss * current_H_frac)  # current_H_frac is 0 by default

    # J_HNO2 = Constant(
    #     J_HNO2_prefactor * current_OHP_ss * (-0.5) * NO2_FE
    # )

    J_NO2 = Constant(
        J_NO2_prefactor * current_OHP_ss * (-0.5) * NO2_FE)

    J_NH4 = Constant(
        J_NH4_prefactor * current_OHP_ss * (-0.125) * NH4_FE)

    # J_NH3 = Constant(
    #     J_NH3_prefactor * current_OHP_ss * (-0.125) * NH4_FE)

    # # STANDARD CASE: if NH4+ and NO2- are formed at cathode:
    J_OH = Constant(
        J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-1.0) * \
        (NO2_FE + H2_FE) + J_OH_prefactor * current_OHP_ss * \
        (1 - current_H_frac) * (-10 / 8) * NH4_FE)
    # for NH4+ production, 8 electrons are consumed for 10 hydroxides produced

    # if HNO2 & NH4+ are getting formed at cathode:
    # J_OH = Constant(
    #     J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-1.0) * H2_FE
    #     + J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-10 / 8) * NH4_FE
    #     + J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-3 / 2) * NO2_FE)

    # if HNO2 & NH3 are getting formed at cathode:
    # J_OH = Constant(
    #     J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-1.0) * H2_FE
    #     + J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-9 / 8) * NH4_FE
    #     + J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-3 / 2) * NO2_FE
    # )

    # storing coordinates of the mesh and the number of vertices
    coor_array = mesh.coordinates()
    num_vertices = mesh.num_vertices()

    # ---------------------------- Homogenous reactions ----------------------------------------------------#

    # R_i are the rates of production of species i (scaled)
    # cation is not being consumed or formed in any homogeneous reaction

    # # Rate constants + hydroxide-based equilibrium:
    kf_NH3 = 6.1*(10**5)  # forward rate constant for NH3 + H2O <--> NH4+ + OH-
    kr_NH3 = kf_NH3/(1.78*(10**(-5)))  # reverse rate constant for NH3 + H2O <--> NH4+ + OH-
    R_NH4 = - scale_R['NH4'] * (-kf_NH3 * initial_conc['NH3'] * u_NH3 + kr_NH3 * initial_conc['NH4'] * u_NH4 *
                              initial_conc['OH'] * u_OH)
    R_NH3 = - scale_R['NH3'] * (kf_NH3 * initial_conc['NH3'] * u_NH3 - kr_NH3 * initial_conc['NH4'] * u_NH4 *
                                initial_conc['OH'] * u_OH)

    kf_NO2 = 1.0e5
    Keq_NO2 = 5.568e10  # equilibrium constant for NO2- + H2O <--> HNO2 + OH-
    kr_NO2 = kf_NO2/Keq_NO2
    R_NO2 = - scale_R['NO2'] * (kf_NO2 * initial_conc['NO2'] * u_NO2 - kr_NO2 * initial_conc['HNO2'] * u_HNO2 *
                                initial_conc['OH'] * u_OH)
    R_HNO2 = - scale_R['HNO2'] * (-kf_NO2 * initial_conc['NO2'] * u_NO2 + kr_NO2 * initial_conc['HNO2'] * u_HNO2 *
                                  initial_conc['OH'] * u_OH)

    R_H = - scale_R['H'] * (
            kw2 * (u_H * initial_conc['H']) * (u_OH * initial_conc['OH']) - kw1
    )

    R_OH = - scale_R['OH'] * (
            kw2 * (u_H * initial_conc['H']) * (u_OH * initial_conc['OH']) - kw1
            - kf_NH3 * initial_conc['NH3'] * u_NH3 + kr_NH3 * initial_conc['NH4'] * u_NH4 * initial_conc['OH'] * u_OH
            - kf_NO2 * initial_conc['NO2'] * u_NO2 + kr_NO2 * initial_conc['HNO2'] * u_HNO2 *
        initial_conc['OH'] * u_OH
    )

    # ------------------------------ Poisson equation expressions -----------------------------------------------#
    F_p = - (
            eps_rel * (
            (55 - (n_water[cat_str] * u_cat * initial_conc[cat_str]
                   + n_water['H'] * u_H * initial_conc['H']) * 1.0e-3) / 55
    )
            + 6 * (
                    ((n_water[cat_str] * u_cat * initial_conc[cat_str]
                      + n_water['H'] * u_H * initial_conc['H']) * 1.0e-3) / 55
            )
    ) * dot(grad(u_p), grad(v_p)) * dx + (
                  z['H'] * u_H * initial_conc['H']
                  + z['OH'] * u_OH * initial_conc['OH']
                  + z['NO3'] * u_NO3 * initial_conc['NO3']
                  + z['NO2'] * u_NO2 * initial_conc['NO2']
                  + z['NH4'] * u_NH4 * initial_conc['NH4']
                  + z['ClO4'] * u_ClO4 * initial_conc['ClO4']
                  + z[cat_str] * u_cat * initial_conc[cat_str]
          ) * q * v_p * dx

    # ------------------------------ Nernst-Planck expressions -----------------------------------------------# For
    # NO3RR, the PNP formulation was ignored. Thus, the modifications to this section are incomplete! So don't run this
    # code using PNP as an input if you're trying to simulate NO3RR!
    if model == 'PNP':
        F_H = ((u_H - u_nH) / (del_t * L_D)) * v_H * dx \
              + dot(grad(u_H), grad(v_H)) * dx \
              + z['H'] * u_H * dot(grad(u_p), grad(v_H)) * dx - R_H * v_H * dx

        F_OH = ((u_OH - u_nOH) / (del_t * L_D)) * v_OH * dx \
               + dot(grad(u_OH), grad(v_OH)) * dx \
               + z['OH'] * u_OH * dot(grad(u_p), grad(v_OH)) * dx \
               - R_OH * v_OH * dx

        F_NO3 = ((u_NO3 - u_nNO3) / (del_t * L_D)) * v_NO3 * dx \
                + dot(grad(u_NO3), grad(v_NO3)) * dx \
                + z['NO3'] * u_NO3 * dot(grad(u_p), grad(v_NO3)) * dx \
                + J_NO3 * v_NO3 * ds

        F_cat = ((u_cat - u_ncat) / (del_t * L_D)) * v_cat * dx \
                + dot(grad(u_cat), grad(v_cat)) * dx \
                + z[cat_str] * u_cat * dot(grad(u_p), grad(v_cat)) * dx

        F = F_H + F_OH + F_NO3 + F_cat + F_p

    # THIS is where all the modifications were done. Be sure to specify "MPNP" as an input when running this code!
    elif model == 'MPNP':

        F_H = (diff_coeff['H'] / diff_coeff['NO3']) * ((u_H - u_nH) / (del_t * L_D)) * v_H * dx \
              + dot(grad(u_H), grad(v_H)) * dx \
              + z['H'] * u_H * dot(grad(u_p), grad(v_H)) * dx \
              - R_H * v_H * dx \
              + J_H * v_H * ds \
              + (
                      u_H / (1 - (
                      scale_vol['H'] * u_H
                      + scale_vol['OH'] * u_OH
                      + scale_vol['NO3'] * u_NO3
                      + scale_vol['NO2'] * u_NO2
                      + scale_vol['HNO2'] * u_HNO2
                      + scale_vol['NH4'] * u_NH4
                      + scale_vol['NH3'] * u_NH3
                      + scale_vol['ClO4'] * u_ClO4
                      + scale_vol[cat_str] * u_cat
              ))
              ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_H)
        ) * dx

        F_OH = (diff_coeff['OH'] / diff_coeff['NO3']) * ((u_OH - u_nOH) / (del_t * L_D)) * v_OH * dx \
               + dot(grad(u_OH), grad(v_OH)) * dx \
               + z['OH'] * u_OH * dot(grad(u_p), grad(v_OH)) * dx \
               - R_OH * v_OH * dx \
               + J_OH * v_OH * ds \
               + (
                       u_OH / (1 - (
                       scale_vol['H'] * u_H
                       + scale_vol['OH'] * u_OH
                       + scale_vol['NO3'] * u_NO3
                       + scale_vol['NO2'] * u_NO2
                       + scale_vol['HNO2'] * u_HNO2
                       + scale_vol['NH4'] * u_NH4
                       + scale_vol['NH3'] * u_NH3
                       + scale_vol['ClO4'] * u_ClO4
                       + scale_vol[cat_str] * u_cat
               ))
               ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_OH)
        ) * dx

        F_NO3 = (diff_coeff['NO3'] / diff_coeff['NO3']) * ((u_NO3 - u_nNO3) / (del_t * L_D)) * v_NO3 * dx \
                + dot(grad(u_NO3), grad(v_NO3)) * dx \
                + z['NO3'] * u_NO3 * dot(grad(u_p), grad(v_NO3)) * dx \
                + J_NO3 * v_NO3 * ds \
                + (
                        u_NO3 / (1 - (
                        scale_vol['H'] * u_H
                        + scale_vol['OH'] * u_OH
                        + scale_vol['NO3'] * u_NO3
                        + scale_vol['NO2'] * u_NO2
                        + scale_vol['HNO2'] * u_HNO2
                        + scale_vol['NH4'] * u_NH4
                        + scale_vol['NH3'] * u_NH3
                        + scale_vol['ClO4'] * u_ClO4
                        + scale_vol[cat_str] * u_cat
                ))
                ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_NO3)
        ) * dx

        # For NO2 and HNO2, need to remove/add J_X term depending on which species we are giving flux BC to
        # Default is to not have J term for HNO2 since NO2- is what is forming at the cathode
        F_NO2 = (diff_coeff['NO2'] / diff_coeff['NO3']) * ((u_NO2 - u_nNO2) / (del_t * L_D)) * v_NO2 * dx \
                + dot(grad(u_NO2), grad(v_NO2)) * dx \
                + z['NO2'] * u_NO2 * dot(grad(u_p), grad(v_NO2)) * dx \
                - R_NO2 * v_NO2 * dx \
                + J_NO2 * v_NO2 * ds \
                + (
                        u_NO2 / (1 - (
                        scale_vol['H'] * u_H
                        + scale_vol['OH'] * u_OH
                        + scale_vol['NO3'] * u_NO3
                        + scale_vol['NO2'] * u_NO2
                        + scale_vol['HNO2'] * u_HNO2
                        + scale_vol['NH4'] * u_NH4
                        + scale_vol['NH3'] * u_NH3
                        + scale_vol['ClO4'] * u_ClO4
                        + scale_vol[cat_str] * u_cat
                ))
                ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_NO2)
        ) * dx

        F_HNO2 = (diff_coeff['HNO2'] / diff_coeff['NO3']) * ((u_HNO2 - u_nHNO2) / (del_t * L_D)) * v_HNO2 * dx \
                 + dot(grad(u_HNO2), grad(v_HNO2)) * dx \
                 - R_HNO2 * v_HNO2 * dx \
                 + (
                         u_NO2 / (1 - (
                         scale_vol['H'] * u_H
                         + scale_vol['OH'] * u_OH
                         + scale_vol['NO3'] * u_NO3
                         + scale_vol['NO2'] * u_NO2
                         + scale_vol['HNO2'] * u_HNO2
                         + scale_vol['NH4'] * u_NH4
                         + scale_vol['NH3'] * u_NH3
                         + scale_vol['ClO4'] * u_ClO4
                         + scale_vol[cat_str] * u_cat
                 ))
                 ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_NO2)
        ) * dx

        F_NH4 = (diff_coeff['NH4'] / diff_coeff['NO3']) * ((u_NH4 - u_nNH4) / (del_t * L_D)) * v_NH4 * dx \
                + dot(grad(u_NH4), grad(v_NH4)) * dx \
                + z['NH4'] * u_NH4 * dot(grad(u_p), grad(v_NH4)) * dx \
                + J_NH4 * v_NH4 * ds \
                - R_NH4 * v_NH4 * dx \
                + (
                        u_NH4 / (1 - (scale_vol['H'] * u_H
                                      + scale_vol['OH'] * u_OH
                                      + scale_vol['NO3'] * u_NO3
                                      + scale_vol['NO2'] * u_NO2
                                      + scale_vol['HNO2'] * u_HNO2
                                      + scale_vol['NH4'] * u_NH4
                                      + scale_vol['NH3'] * u_NH3
                                      + scale_vol['ClO4'] * u_ClO4
                                      + scale_vol[cat_str] * u_cat))
                ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_NH4)
        ) * dx

        F_NH3 = (diff_coeff['NH3'] / diff_coeff['NO3']) * ((u_NH3 - u_nNH3) / (del_t * L_D)) * v_NH3 * dx \
                + dot(grad(u_NH3), grad(v_NH3)) * dx \
                - R_NH3 * v_NH3 * dx \
                + (
                        u_NH3 / (1 - (
                        scale_vol['H'] * u_H
                        + scale_vol['OH'] * u_OH
                        + scale_vol['NO3'] * u_NO3
                        + scale_vol['NO2'] * u_NO2
                        + scale_vol['HNO2'] * u_HNO2
                        + scale_vol['NH4'] * u_NH4
                        + scale_vol['NH3'] * u_NH3
                        + scale_vol['ClO4'] * u_ClO4
                        + scale_vol[cat_str] * u_cat
                ))
                ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_NH3)
        ) * dx

        F_ClO4 = (diff_coeff['ClO4'] / diff_coeff['NO3']) * ((u_ClO4 - u_nClO4) / (del_t * L_D)) * v_ClO4 * dx \
                 + dot(grad(u_ClO4), grad(v_ClO4)) * dx \
                 + z['ClO4'] * u_ClO4 * dot(grad(u_p), grad(v_ClO4)) * dx \
                 + (
                         u_ClO4 / (1 - (
                         scale_vol['H'] * u_H
                         + scale_vol['OH'] * u_OH
                         + scale_vol['NO3'] * u_NO3
                         + scale_vol['NO2'] * u_NO2
                         + scale_vol['HNO2'] * u_HNO2
                         + scale_vol['NH4'] * u_NH4
                         + scale_vol['NH3'] * u_NH3
                         + scale_vol['ClO4'] * u_ClO4
                         + scale_vol[cat_str] * u_cat
                 ))
                 ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_ClO4)
        ) * dx

        F_cat = (diff_coeff[cat_str] / diff_coeff['NO3']) * ((u_cat - u_ncat) / (del_t * L_D)) * v_cat * dx \
                + dot(grad(u_cat), grad(v_cat)) * dx \
                + z[cat_str] * u_cat * dot(grad(u_p), grad(v_cat)) * dx \
                + (
                        u_cat / (1 - (
                        scale_vol['H'] * u_H
                        + scale_vol['OH'] * u_OH
                        + scale_vol['NO3'] * u_NO3
                        + scale_vol['NO2'] * u_NO2
                        + scale_vol['HNO2'] * u_HNO2
                        + scale_vol['NH4'] * u_NH4
                        + scale_vol['NH3'] * u_NH4
                        + scale_vol['ClO4'] * u_ClO4
                        + scale_vol[cat_str] * u_cat
                ))
                ) * dot(
            scale_vol['H'] * grad(u_H)
            + scale_vol['OH'] * grad(u_OH)
            + scale_vol['NO3'] * grad(u_NO3)
            + scale_vol['NO2'] * grad(u_NO2)
            + scale_vol['NH4'] * grad(u_NH4)
            + scale_vol['NH3'] * grad(u_NH3)
            + scale_vol['ClO4'] * grad(u_ClO4)
            + scale_vol['HNO2'] * grad(u_HNO2)
            + scale_vol[cat_str] * grad(u_cat),
            grad(v_cat)
        ) * dx

        F = F_H + F_OH + F_NO3 + F_NO2 + F_NH4 + F_ClO4 + F_NH3 + F_cat + F_p + F_HNO2

    # not bothering with stabilization modification for NO3RR because it does not apply to MPNP equations
    if stabilization == 'Y':
        # normalised over the system length
        h = project(CellDiameter(mesh)).compute_vertex_values()

        rho = {
            'H': np.zeros(num_vertices), 'OH': np.zeros(num_vertices),
            'NO3': np.zeros(num_vertices), 'NO2': np.zeros(num_vertices),
            'NH4': np.zeros(num_vertices), cat_str: np.zeros(num_vertices)
        }

        Pe = {
            'H': np.zeros(num_vertices), 'OH': np.zeros(num_vertices),
            'NO3': np.zeros(num_vertices), 'NO2': np.zeros(num_vertices),
            'NH4': np.zeros(num_vertices), cat_str: np.zeros(num_vertices)
        }

        rho_large = {
            'H': np.zeros(num_vertices), 'OH': np.zeros(num_vertices),
            'NO3': np.zeros(num_vertices), 'NO2': np.zeros(num_vertices),
            'NH4': np.zeros(num_vertices), cat_str: np.zeros(num_vertices)
        }

        fact = 1
        # rho_small = h
        rho_small = fact ** 2 * h ** 2 / 4  # value of rho if Pe is <=1

    H = np.ones(num_vertices)
    OH = np.ones(num_vertices)
    NO3 = np.ones(num_vertices)
    NO2 = np.ones(num_vertices)
    HNO2 = np.ones(num_vertices)
    NH4 = np.ones(num_vertices)
    NH3 = np.ones(num_vertices)
    ClO4 = np.ones(num_vertices)
    cat = np.ones(num_vertices)
    p = np.zeros(num_vertices)

    # Time-stepping
    t = 0
    for n in range(tot_num_steps):

        if dry_run:
            print(int(t / dt))
        else:
            # Update current time
            if t >= T_1:
                # choose the larger time step beyond T_1
                dt = dts[1]
                del_t = del_ts[1]
                print(int(t / dt))
                # adjust number of steps
                # print(int(num_steps_1 + (t - T_1) / dt))
            # elif t >= T_2:
            #     # choose the larger time step beyond T_1
            #     dt = dts[2]
            #     del_t = del_ts[2]
            else:
                print(int(t / dt))

        t += dt

        if stabilization == 'Y':
            # norm of the gradient of potential projected on mesh
            norm_grad_phi = project(
                sqrt(inner(grad(u_np), grad(u_np)))
            ).compute_vertex_values()

            for specie in species:
                if z[specie] != 0:
                    Pe[specie] = \
                        (fact * h * norm_grad_phi * abs(z[specie])) / 2

                    rho_large[specie] = \
                        fact * h / (2 * abs(z[specie]) * norm_grad_phi)
                    # rho_large[specie] = h

                    # check if Pe number of > or <= 1
                    for n in range(num_vertices):
                        if Pe[specie][n] > 1.0 + tol:
                            rho[specie][n] = rho_large[specie][n]
                        else:
                            rho[specie][n] = rho_small[n]
                else:
                    continue

            # defining functions over scalar function space Y
            rho_H = Function(Y)
            rho_OH = Function(Y)
            rho_NO3 = Function(Y)
            rho_NO2 = Function(Y)
            rho_NH4 = Function(Y)
            rho_NH3 = Function(Y)
            rho_cat = Function(Y)

            rho_H.vector().set_local(np.flip(rho['H'], 0))
            rho_OH.vector().set_local(np.flip(rho['OH'], 0))
            rho_NO3.vector().set_local(np.flip(rho['NO3'], 0))
            rho_NO2.vector().set_local(np.flip(rho['NO2'], 0))
            rho_NH4.vector().set_local(np.flip(rho['NH4'], 0))
            rho_NH3.vector().set_local(np.flip(rho['NH3'], 0))
            rho_cat.vector().set_local(np.flip(rho[cat_str], 0))

            if model == 'PNP':

                F_stab_PNP = -1.0 * (
                        rho_H * z['H'] * (
                        (u_H - u_nH) / (del_t * L_D)
                        + z['H'] * dot(grad(u_H), grad(u_p))
                        + R_H
                ) * dot(grad(u_p), grad(v_H)) * dx
                        + rho_OH * z['OH'] * (
                                (u_OH - u_nOH) / (del_t * L_D)
                                + z['OH'] * dot(grad(u_H), grad(u_p))
                                + R_OH
                        ) * dot(grad(u_p), grad(v_OH)) * dx
                        + rho_NO3 * z['NO3'] * (
                                (u_NO3 - u_nNO3) / (del_t * L_D)
                                + z['NO3'] * dot(grad(u_NO3), grad(u_p))
                        ) * dot(grad(u_p), grad(v_NO3)) * dx
                        + rho_cat * z[cat_str] * (
                                (u_cat - u_ncat) / (del_t * L_D)
                                + z[cat_str] * dot(grad(u_cat), grad(u_p))
                        ) * dot(grad(u_p), grad(v_cat)) * dx
                )

                # Solve variational problem for time step
                solve(
                    F + F_stab_PNP + J_OH * v_OH * ds + J_H * v_H * ds == 0,
                    u,
                    bcs,
                    solver_parameters=solver_parameters
                )

            elif model == 'MPNP':
                # SUPG stabilization formulation not well suited for MPNP due to volume term. Do not use it.
                print("Warning:stabilization not implemented for MPNP!")

                solve(
                    F + J_OH * v_OH * ds + J_H * v_H * ds == 0,
                    u,
                    bcs,
                    solver_parameters=solver_parameters
                )

        else:
            solve(
                F == 0,
                u,
                bcs,
                solver_parameters=solver_parameters
            )

        # Save solution to file (VTK)
        _u_H, _u_OH, _u_NO3, _u_NO2, _u_NH4, _u_NH3, _u_ClO4, _u_HNO2, _u_cat, _u_p = u.split()

        _u_H_nodal_values_array = _u_H.compute_vertex_values()
        _u_OH_nodal_values_array = _u_OH.compute_vertex_values()
        _u_NO3_nodal_values_array = _u_NO3.compute_vertex_values()
        _u_NO2_nodal_values_array = _u_NO2.compute_vertex_values()
        _u_HNO2_nodal_values_array = _u_HNO2.compute_vertex_values()
        _u_NH4_nodal_values_array = _u_NH4.compute_vertex_values()
        _u_NH3_nodal_values_array = _u_NH3.compute_vertex_values()
        _u_ClO4_nodal_values_array = _u_ClO4.compute_vertex_values()
        _u_cat_nodal_values_array = _u_cat.compute_vertex_values()
        _u_p_nodal_values_array = _u_p.compute_vertex_values()

        # creating a numpy array of concentration values
        # at every time step in the whole domain
        H = np.vstack((H, _u_H_nodal_values_array))
        OH = np.vstack((OH, _u_OH_nodal_values_array))
        NO3 = np.vstack((NO3, _u_NO3_nodal_values_array))
        NO2 = np.vstack((NO2, _u_NO2_nodal_values_array))
        HNO2 = np.vstack((HNO2, _u_HNO2_nodal_values_array))
        NH4 = np.vstack((NH4, _u_NH4_nodal_values_array))
        NH3 = np.vstack((NH3, _u_NH3_nodal_values_array))
        ClO4 = np.vstack((ClO4, _u_ClO4_nodal_values_array))
        cat = np.vstack((cat, _u_cat_nodal_values_array))
        p = np.vstack((p, _u_p_nodal_values_array))

        # fraction (wrt bulk) of protons at the OHP
        H_OHP_frac = _u_H_nodal_values_array[0]

        # adjust proton consumption current iteratively
        # if H_OHP frac is higher than a specified limit
        if H_OHP is not None:
            if H_OHP_frac < 0:
                current_H_frac = current_H_frac / 1.1
            elif H_OHP_frac < (H_OHP - 0.05):
                current_H_frac = current_H_frac / 1.05
            elif H_OHP_frac < (H_OHP - 0.025):
                current_H_frac = current_H_frac / 1.01
            elif (H_OHP_frac > H_OHP and
                  H_OHP_frac <= (H_OHP + 0.4) and
                  current_H_frac <= 1.0):
                current_H_frac = current_H_frac * 1.04
            elif H_OHP_frac > (H_OHP + 0.4) and current_H_frac <= 1.0:
                current_H_frac = current_H_frac * 1.15

            # print(H_OHP_frac)
            # print(current_H_frac)

            # new flux of OH and H after adjustment of
            # the proton consumption current at OHP
            # J_OH = Constant(
            #     -1.0 * J_OH_prefactor * current_OHP_ss * (1 - current_H_frac)
            # )


            J_OH = Constant(
                J_OH_prefactor * current_OHP_ss * (1 - current_H_frac) * (-1.0) * \
                (NO2_FE + H2_FE) + J_OH_prefactor * current_OHP_ss * \
                (1 - current_H_frac) * (-10 / 8) * NH4_FE
                # for NH4+ production, 8 electrons are consumed for 10 hydroxides produced
            )

            J_H = Constant(J_H_prefactor * current_OHP_ss * current_H_frac)

        # Update previous solution
        u_n.assign(u)

    end_time = datetime.now().strftime('%y-%m-%d-%H-%M-%S')

    # estimating the electric field value as a function
    # of x for the last computed value of potential profile
    field = project(-grad(u_np), W)  # pylint: disable=E1130
    field_values = field.compute_vertex_values()
    field_values_rescaled = field_values * thermal_voltage / L_n
    field_OHP = field_values_rescaled[0] * 1.0e-9  # in V/nm

    if dry_run:
        # time points as array without staging
        tau_array = np.linspace(0, T, tot_num_steps)
    else:
        # with staging
        tau_array_1 = np.linspace(0, T_1, num_steps_1)
        tau_array_2 = np.linspace(T_1 + dts[1], T_2, num_steps_2)
        # tau_array_3 = np.linspace(T_2 + dts[2], T_3, num_steps_3)
        # concatenate the two time step arrays
        tau_array = np.concatenate((tau_array_1, tau_array_2))

    if stabilization != 'Y':
        Pe = None
        rho = None

    np.savez(
        newpath + '/arrays_unscaled.npz',
        H=H,
        OH=OH,
        NO3=NO3,
        NO2=NO2,
        HNO2=HNO2,
        NH4=NH4,
        NH3=NH3,
        ClO4=ClO4,
        cat=cat,
        p=p,
        coor=coor_array,
        tau=tau_array,
        field_values=field_values)

    # rescaling the output. all outputs in SI unitsand numpy arrays
    t_H, c_H = scale(
        species='H',
        tau=tau_array,
        C=H,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_OH, c_OH = scale(
        species='OH',
        tau=tau_array,
        C=OH,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_NO3, c_NO3 = scale(
        species='NO3',
        tau=tau_array,
        C=NO3,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_NO2, c_NO2 = scale(
        species='NO2',
        tau=tau_array,
        C=NO2,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_HNO2, c_HNO2 = scale(
        species='HNO2',
        tau=tau_array,
        C=HNO2,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_NH4, c_NH4 = scale(
        species='NH4',
        tau=tau_array,
        C=NH4,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_NH3, c_NH3 = scale(
        species='NH3',
        tau=tau_array,
        C=NH3,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_ClO4, c_ClO4 = scale(
        species='ClO4',
        tau=tau_array,
        C=ClO4,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    t_cat, c_cat = scale(
        species=cat_str,
        tau=tau_array,
        C=cat,
        initial_conc=initial_conc,
        L_n=L_n,
        L_debye=L_debye)

    coor_scaled = coor_array * L_n

    psi = p * thermal_voltage

    # Relative permittivity at OHP, not used unless you're doing Stern calc. Modify with cations in sim.
    eps_rel_conc_ss = \
        eps_rel * ((55 - (n_water[cat_str] * c_cat
                          + n_water['H'] * c_H) * 1.0e-3) / 55) \
        + 6 * (((n_water[cat_str] * c_cat + n_water['H'] * c_H) * 1.0e-3) / 55)

    eps_rel_OHP = eps_rel_conc_ss[-1][0]

    # at steady state as a function of x
    charge_density = \
        c_cat[-1] - c_NO3[-1] - c_NO2[-1] + c_NH4[-1] + c_OH[-1] + c_H[-1] + c_ClO4[-1]

    np.savez(
        newpath + '/arrays_scaled.npz',
        x=coor_scaled,
        psi=psi,
        t_H=t_H,
        c_H=c_H,
        t_OH=t_OH,
        c_OH=c_OH,
        t_NO3=t_NO3,
        c_NO3=c_NO3,
        t_NO2=t_NO2,
        t_HNO2=t_HNO2,
        c_HNO2=c_HNO2,
        c_NO2=c_NO2,
        t_NH4=t_NH4,
        t_NH3=t_NH3,
        c_NH3=c_NH3,
        c_NH4=c_NH4,
        t_ClO4=t_ClO4,
        c_ClO4=c_ClO4,
        t_cat=t_cat,
        c_cat=c_cat,
        eps_rel=eps_rel_conc_ss,
        field_values=field_values_rescaled,
        charge_density=charge_density)

    # initiating empty lists for storing OHP concentration values (only necessary if doing Stern layer calculation)
    H_surf = []
    OH_surf = []
    NO3_surf = []
    NO2_surf = []
    HNO2_surf = []
    NH4_surf = []
    NH3_surf = []
    cat_surf = []
    p_surf = []

    for i in range(0, len(t_cat)):
        H_surf += [c_H[i][0]]
        OH_surf += [c_OH[i][0]]
        NO3_surf += [c_NO3[i][0]]
        NO2_surf += [c_NO2[i][0]]
        HNO2_surf += [c_HNO2[i][0]]
        NH4_surf += [c_NH4[i][0]]
        NH3_surf += [c_NH3[i][0]]
        cat_surf += [c_cat[i][0]]
        p_surf += [psi[i][0]]

    potential_OHP = p_surf[-1]

    NO3_OHP_frac = NO3_surf[-1] / initial_conc['NO3']

    print(NO3_OHP_frac)

    # current density attributed to proton consumption
    current_H = current_H_frac * current_OHP_ss

    # Uncomment next two lines for long time, script will break if it is uncommented for dry runs:
    time_step = time_step_1  # if staged
    total_sim_time = total_sim_time_2  # if time is staged

    # create and open metadata file
    f_meta = open(newpath + '/metadata.json', 'w')

    metadata_dict = {
        'concentration_elec': concentration_elec,
        'cation': cation,
        'model': model,
        'stabilization': stabilization,
        'voltage_multiplier': voltage_multiplier,
        'H2_FE': H2_FE,
        'NO2_FE': NO2_FE,
        'NH4_FE': NH4_FE,
        'L_n_EDL': L_n,
        'time_constant': time_constant,
        'time_step': time_step,
        'total_sim_time': total_sim_time,
        'mesh_number': mesh_number,
        'mesh_structure': mesh_structure,
        'eps_rel_OHP': eps_rel_OHP,
        'field_OHP': field_OHP,
        'current_OHP_ss': current_OHP_ss,
        'current_H': current_H,
        'H_OHP_vs_bulk': H_OHP,
        'potential_OHP': potential_OHP,
        # 'pH_OHP': pH_OHP,
        'NO3_OHP_frac': NO3_OHP_frac,
        # 'pH_overpotential': pH_overpotential,
        # 'NO3_overpotential': NO3_overpotential,
        'end_time': end_time}

    r = json.dumps(metadata_dict, indent=0)
    f_meta.write(r)
    f_meta.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment parameters')

    parser.add_argument(
        '--concentration_elec',
        metavar='electrolyte_concentration',
        required=False,
        help='float val, 0.1 M',
        default=1.0,
        type=float
    )

    parser.add_argument(
        '--model',
        metavar='model_type',
        required=False,
        help='str, PNP/MPNP',
        default='MPNP',
        type=str
    )

    parser.add_argument(
        '--voltage_multiplier',
        metavar='thermal_voltage_multiplier',
        required=False,
        help='float val, -1.0',
        default=-4.0,
        type=float
    )

    parser.add_argument(
        '--mesh_structure',
        metavar='bias in mesh structure',
        required=False,
        help='str, uniform/variable',
        default='variable',
        type=str
    )

    parser.add_argument(
        '--H2_FE',
        metavar='faradaic efficiency for hydrogen in fraction',
        required=False,
        help='float val, 0.2',
        default=0.201,
        type=float
    )

    parser.add_argument(
        '--NO2_FE',
        metavar='faradaic efficiency for NO2- in fraction',
        required=False,
        help='float val, 0.2',
        default=0.1898,
        type=float
    )

    parser.add_argument(
        '--current_OHP_ss',
        metavar='steady state current in A/m2',
        required=False,
        help='float val, 10.0',
        default=14.1,
        type=float
    )

    parser.add_argument(
        '--L_n',
        metavar='system size',
        required=False,
        help='float val, 50.0e-6',
        default=217.0e-6,
        type=float
    )

    parser.add_argument(
        '--stabilization',
        metavar='SUPG',
        required=False,
        help='str, Y/N',
        default='N',
        type=str
    )

    parser.add_argument(
        '--H_OHP',
        metavar='build up of protons at the OHP relative to the bulk',
        required=False,
        help='float val, None/1.1/2.0',
        default=None,
        type=float
    )

    parser.add_argument(
        '--cation',
        metavar='monovalent cation in solution',
        required=False,
        help='str, K/Cs/Li',
        default='Na',
        type=str
    )

    parser.add_argument(
        '--params_file',
        metavar='yaml file with parameter values',
        required=False,
        help='str, parameters',
        default='parameters',
        type=str
    )

    parser.add_argument(
        '--dry_run',
        metavar='run 100 time steps as test',
        required=False,
        help='boolean value',
        default=False,
        type=bool
    )

    args = parser.parse_args()

    solve_EDL(
        concentration_elec=args.concentration_elec,
        model=args.model,
        voltage_multiplier=args.voltage_multiplier,
        mesh_structure=args.mesh_structure,
        H2_FE=args.H2_FE,
        NO2_FE=args.NO2_FE,
        current_OHP_ss=args.current_OHP_ss,
        L_n=args.L_n,
        stabilization=args.stabilization,
        H_OHP=args.H_OHP,
        cation=args.cation,
        params_file=args.params_file,
        dry_run=args.dry_run
    )
