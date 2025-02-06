import numpy as np
import matplotlib.pyplot as plt

def run_long_parallel_simulations(propagator, system, kT, x_init_coord, dt, nsteps, save_period, n_parallel):
    
    #---------------------------------------------------------------
    #run n_parallel long simulations
    
    #reinitialize x_init each time
    x_init = np.array([x_init_coord for element in range(n_parallel)])
    long_trjs = np.array(propagator(system, kT, x_init, dt, nsteps, save_period))

    return long_trjs


def estimate_energy_landscape_histogram(trjs, kT, nbins, binrange = [], symmetric = True):
    
    #-------define bins--------------------------------------------------------------------

    #flatten trajectory since the order of the frames does not matter here
    trj_flat = trjs.flatten()

    #set bin boundaries

    if binrange == []:

        epsilon = 10**-9 #to avoid any >= vs > issues at bin boundaries
        if symmetric:    
            bin_extreme = max(np.max(trj_flat), -np.min(trj_flat))
            bin_min = -bin_extreme-epsilon
            bin_max = bin_extreme+epsilon
        else:
            bin_min = np.min(trj_flat)-epsilon
            bin_max = np.max(trj_flat)+epsilon
    else:
        bin_min = binrange[0]
        bin_max = binrange[1]

    step = (bin_max-bin_min)/nbins
    
    binbounds = np.linspace(bin_min, bin_max, nbins+1)
    bincenters = np.linspace(bin_min-step/2, bin_max+step/2, nbins+2)

    #-------bin trajectory--------------------------------------------------------------------

    #note that digitize reserves the index 0 for entries below the first bin, 
    # but in this case the bins have been constructed so that all entries lie within the explicit bin range
    # so the first and last bins of eq_pops should be empty
    binned_trj = np.digitize(trj_flat, bins = binbounds)

    eq_pops = np.zeros(nbins+2)
    for b in binned_trj:
        eq_pops[b] += 1/len(trj_flat)

    #-------calculate energies--------------------------------------------------------------------
    #return only the continuous central range where all bins have nonzero occupancy

    energies = []
    ind_min = 0
    ind_max = nbins+1
    x_sampled = []
    e_sampled = []

    #calculate energies and identify the area of continuous sampling
    for i, p in enumerate(eq_pops):
        if p != 0:
            #normalized probability to calculate energy per unit x rather than per bin
            energies.append(-kT*np.log(p/step))

            x_sampled.append(bincenters[i])
            e_sampled.append(energies[-1])
        else:
            energies.append(0)
            if i < (nbins+2)/2:
                ind_min = i+1
            elif ind_max == nbins+1:
                ind_max = i
            

    #all sampled points
    x_cont = bincenters[ind_min:ind_max]
    e_cont = energies[ind_min:ind_max]
    
    return bincenters, eq_pops, x_sampled, e_sampled, x_cont, e_cont


#calculate mean first passage time from long trajectories
def calc_mfpt(macrostate_classifier, n_macrostates, save_period, trajectories):

    n_transitions = np.zeros([n_macrostates, n_macrostates])
    frames_by_state = np.zeros(n_macrostates)
    
    for trj in trajectories.transpose():
        
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
    