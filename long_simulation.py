import numpy as np
import matplotlib.pyplot as plt
import MSM_methods

def run_long_parallel_simulations(propagator, system, kT, x_init_coord, dt, nsteps, save_period, n_parallel):
    
    #---------------------------------------------------------------
    #run n_parallel long simulations
    
    #reinitialize x_init each time
    x_init = np.array([x_init_coord for element in range(n_parallel)])
    long_trjs = np.array(propagator(system, kT, x_init, dt, nsteps, save_period))

    return long_trjs


def get_bin_boundaries(trj_flat, nbins, binrange = [], symmetric = True):
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

    return binbounds, bincenters, step


def estimate_energy_landscape_histogram(trjs, kT, nbins, binrange = [], symmetric = True):
    

    #flatten trajectory since the order of the frames does not matter here
    trj_flat = trjs.flatten()

    #-------define bins--------------------------------------------------------------------

    binbounds, bincenters, step = get_bin_boundaries(trj_flat, nbins, binrange, symmetric)

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
    


def msm_analysis(trjs, kT, nbins, macrostate_classifier, n_macrostates, save_period, binrange = [], symmetric = True, show_TPM=False):

    #get bin boundaries
    trj_flat = trjs.flatten()
    binbounds, bincenters, step = get_bin_boundaries(trj_flat, nbins, binrange, symmetric)

    #-------build MSM--------------------------------------------------------

    trj_discrete = np.digitize(trjs, bins = binbounds)
    transitions = [[trj_discrete[i][j], trj_discrete[i+1][j]] for j in range(trj_discrete.shape[1]) for i in range(len(trj_discrete)-1) ]

    tpm, states_in_order = MSM_methods.transitions_2_msm(transitions)
    if show_TPM:
        plt.imshow(tpm)
        plt.show()

    eqp_msm = MSM_methods.tpm_2_eqprobs(tpm)
    x_msm = [bincenters[i] for i in states_in_order]


    #this part should be abstracted out into MSM_methods
    msm_state_macrostates = [macrostate_classifier(x) for x in x_msm]

    mfpts = np.zeros([n_macrostates, n_macrostates])

    #for each destination macrostate
    for mf in range(n_macrostates):
        transitions_blotted = np.array(tpm)
        for si, s in enumerate(msm_state_macrostates):
            if s == mf:
                transitions_blotted[si,:] = 0
                #transitions_blotted[si,si] = 1

        # plt.imshow(transitions_blotted)
        # plt.show()

        for mi in range(n_macrostates):
            if mi != mf:
                #print(msm_state_macrostates)
                #print()
                eqp_msm_init = np.array([eqpi[0] if msm_state_macrostates[i] == mi else 0 for i, eqpi in enumerate(eqp_msm)]).reshape([len(eqp_msm), 1])
                #for s, si in enumerate(msm_state_macrostates):
                #print(eqp_msm_init)

                p_t = []
                for t in range(1000):
                    p_t.append(np.sum(eqp_msm_init))
                    eqp_msm_init = np.matmul(transitions_blotted, eqp_msm_init)
                    if p_t[-1] <= p_t[0]/2:
                        p_t.append(np.sum(eqp_msm_init))
                        mfpts[mf][mi] = t*save_period
                        break

                # plt.plot(p_t)
                # plt.show()
                

    return x_msm, eqp_msm, msm_state_macrostates, mfpts

