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

    # binned_data = np.digitize(np.array(recorded_positions).flatten(), bins)
    # bin_counts = [np.count(binned_data, i) for i in range(nbins)]

    data_flat = np.array(recorded_positions).flatten()
    histbinned = plt.hist(data_flat, bins)
    plt.show()

    z_kT = sum([np.exp(-system.potential(x)/kT) for x in bincenters])
    eq_pops_analytic = [np.exp(-system.potential(x)/kT)/z_kT for x in bincenters]
    plt.plot(bincenters, eq_pops_analytic)

    eq_pops_simulation = histbinned[0]/len(data_flat)
    plt.plot(bincenters, eq_pops_simulation)
    
    plt.show()

    rmse_weighted = np.sqrt(np.mean([epa*(eps-epa)**2 for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)]))
    kl_divergence = sum([epa*np.log(epa/eps) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])
    print(f"kl divergence = {kl_divergence}")
    print(f"weighted RMSE = {rmse_weighted}")


    
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
    #plt.plot(x_data, e_data)
    
    return x_data, e_data, long_trjs


#calculate mean first passage time from long trajectories
def calc_mfpt(macrostate_classifier, n_macrostates, save_period, trajectories):

    n_transitions = np.zeros([n_macrostates, n_macrostates])
    frames_by_state = np.zeros(n_macrostates)
    
    for trj in trajectories:
        
        #get initial state
        last_state = macrostate_classifier(trj[0])
        last_macrostate = last_state
        for xt in trj:
    
            current_state = macrostate_classifier(xt)

            #the former and latter or conditions describe the following respectively:
            #    transitions between different states regardless of whether any intermediate states were recorded
            #    self transitions in which the trajectory exited the macrostate but returned before reaching a different macrostate
            if (current_state != -1 and last_macrostate != -1) and \
                (current_state != last_macrostate or (current_state == last_macrostate and last_state == -1)):
                n_transitions[current_state][last_macrostate] += 1           

            #update buffers
            last_state = current_state
            if current_state != -1:
                frames_by_state[current_state] += 1
                last_macrostate = current_state

    n_steps = sum([len(trj) for trj in trajectories])

    #in steps
    mfpts = save_period*np.reciprocal(n_transitions)*frames_by_state
    
    return n_transitions, mfpts
    