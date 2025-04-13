#propagators.py
#Jonathan Borowsky
#2/21/25

#Method for generating trajectories given a system (potential), temperature, and time step

################################################################################################################

import numpy as np

#parameters
#   trj_coords: list of floats: initial coordinates of the trajectories on the progress coordinate
#   F: function of a float returning a float: 
#      the negative derivative of free energy function with respect to the progress coordinate as a function of the progress coordinate
#   D: float: brownian diffusion coefficient
#   kT: float: Boltzmann's constant times the temperature
#   timestep: float: the size of the timestep used for propagation
#   nsteps: nonnegative int: how many time steps to propagate for

#returns
#   trj_out: list of arrays: the coordinates of the trajectories at each time step
#      trj_out has size [nsteps//save_period, trj_coords.shape[0], trj_coords.shape[1]]

#Brownian diffusion
#nsteps must be an integer multiple of save_period
def propagate(system, kT, trj_coords, timestep, nsteps, save_period):
  
    nd = np.array(trj_coords.shape)   
    D = system.diffusion_coefficient
    
    trj_out = []
    for i in range(nsteps//save_period):
    
        for step in range(save_period):
            trj_coords += D/kT * system.F(trj_coords) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

        trj_out.append(trj_coords.copy())

    return trj_out


#same as propagate() but outputs only the last frame; 
#  avoids an extra layer of for loops when running WE
def propagate_save1(system, kT, trj_coords, timestep, nsteps):
  
    nd = np.array(trj_coords.shape)   
    D = system.diffusion_coefficient
    
    for step in range(nsteps):
        trj_coords += D/kT * system.F(trj_coords) * timestep + np.sqrt(2*D*timestep)*np.random.normal(size=nd)

    return trj_coords


#TODO make a propagator version that actually uses kT for testing replica exchange schemes
def propagate_msm(system_dtmsm, kT, trj_coords, timestep, nsteps, save_period):

    tia = []
    tca = []

    for tci in trj_coords:
        dt_trj = system_dtmsm.dtmsm.simulate(nsteps, start = tci, dt = save_period) #, seed=0
        tia.append(dt_trj)
        tca.append(np.stack([system_dtmsm.x[dti] for dti in dt_trj]))

    return np.stack(tca, axis=0), np.stack(tia, axis=0)