import numpy as np
import matplotlib.pyplot as plt

def recover_energy_landscape(propagator, system, kT, x_init_coord, dt, nsteps, save_period, n_parallel, nbins):
    
    #---------------------------------------------------------------
    #determine probability distribution of n_parallel long simulations
    
    #reinitialize x_init each time
    x_init = np.array([x_init_coord for element in range(n_parallel)])
    long_trjs = np.array(propagator(system, kT, x_init, dt, nsteps, save_period))
    # recorded_positions = []

    # for i in range(nrounds):
    #     x_init = propagator(x_init, prop_params[0], prop_params[1], prop_params[2], prop_params[3], prop_params[4])
    #     #propagate(x_init, F, D, kT, dt, nsteps)
    #     recorded_positions.append(x_init)
        
    #---------------------------------------------------------------
    #examine the probability distribution and bin the trajectory using a histogram
    recorded_positions = long_trjs.flatten()
    
    bin_extreme = max(np.max(recorded_positions), -np.min(recorded_positions))
    #nbins = 101 #polynomial fit is insensitive to this [verify]

    step = 2*bin_extreme/nbins
    
    bins = np.linspace(-bin_extreme, bin_extreme, nbins)
    bincenters = np.linspace(-bin_extreme+step/2, bin_extreme-step/2, nbins-1)
    
    histbinned = plt.hist(np.array(recorded_positions).flatten(), bins)
    plt.show()
    
    #---------------------------------------------------------------
    #calculate and plot the energy per unit x from the probability

    energies = []
    ind_extreme = 1
    ind_extreme_pad = 0

    #calculate energies and identify the useful data range
    for i, p in enumerate(histbinned[0]):
        if p != 0:
            #normalized probability to calculate energy per unit x rather than per bin
            energies.append(-kT*np.log(p/(len(np.array(recorded_positions).flatten())*step)))
        else:
            energies.append(0)
            if i < len(histbinned[0])/2:
                ind_extreme = i+1
            else:
                ind_extreme = max(ind_extreme, len(histbinned[0])-i)

    ind_extreme += ind_extreme_pad

    #plot and fit to the sampled region

    x_data = bincenters[ind_extreme:-ind_extreme]
    e_data = energies[ind_extreme:-ind_extreme]
    plt.plot(x_data, e_data)
    
    return x_data, e_data, long_trjs