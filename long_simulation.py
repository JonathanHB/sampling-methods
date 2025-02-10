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

    #n_steps = sum([len(trj) for trj in trajectories])

    #in steps
    mfpts = save_period*np.reciprocal(n_transitions)*frames_by_state
    
    return n_transitions, mfpts
    


def msm_analysis(trjs, kT, nbins, macrostate_classifier, n_macrostates, save_period, lag_time=1, binrange = [], symmetric = True, show_TPM=False):

    #get bin boundaries
    trj_flat = trjs.flatten()
    binbounds, bincenters, step = get_bin_boundaries(trj_flat, nbins, binrange, symmetric)

    #-------build MSM--------------------------------------------------------

    trj_discrete = np.digitize(trjs, bins = binbounds)
    transitions = [[trj_discrete[i][j], trj_discrete[i+lag_time][j]] for j in range(trj_discrete.shape[1]) for i in range(len(trj_discrete)-lag_time) ]

    tpm, states_in_order = MSM_methods.transitions_2_msm(transitions)
    if show_TPM:
        plt.matshow(tpm)
        plt.show()

    eqp_msm_init = MSM_methods.tpm_2_eqprobs(tpm)
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

        #plt.matshow(transitions_blotted)
        #plt.show()
        #print(transitions_blotted)

        for mi in range(n_macrostates):
            if mi != mf:
                #print(msm_state_macrostates)
                #print()
                eqp_msm = np.array([eqpi[0] if msm_state_macrostates[i] == mi else 0 for i, eqpi in enumerate(eqp_msm_init)]).reshape([len(eqp_msm_init), 1])
                #for s, si in enumerate(msm_state_macrostates):

                rate_per_unit = []
                p_t = [np.sum(eqp_msm)]
                for t in range(500):
                    eqp_msm = np.matmul(transitions_blotted, eqp_msm)
                    p_t.append(np.sum(eqp_msm))
                    prob_in_init = sum([eqpi[0] if msm_state_macrostates[i] == mi else 0 for i, eqpi in enumerate(eqp_msm)]) #there's a faster way of doing this
                    rate_per_unit.append((p_t[-2]-p_t[-1])/prob_in_init)
                    #median first passage time; NOT the MFPT
                    # if p_t[-1] <= p_t[0]/2: 
                    #     p_t.append(np.sum(eqp_msm_init))
                    #     mfpts[mf][mi] = t*save_period
                    #     break
                    #     pass
                mfpts[mf][mi] = save_period*lag_time/rate_per_unit[-1]
                #plt.plot(rate_per_unit)
                #plt.show()
                

    return x_msm, eqp_msm_init, msm_state_macrostates, mfpts



#calculate mean first passage time from long trajectories
def hamsm_analysis(trjs, nbins, system, save_period, lag_time=1, binrange = [], symmetric = True, show_TPM=False):

    nm = system.n_macrostates()

    #get bin boundaries
    trj_flat = trjs.flatten()
    binbounds, bincenters, step = get_bin_boundaries(trj_flat, nbins, binrange, symmetric)

    trjs_discrete = np.digitize(trjs, bins = binbounds).transpose()
    macrostates_discrete = [system.macro_class(x) for x in bincenters]

    #n_transitions = np.zeros([system.n_macrostates*len(bincenters), system.n_macrostates*len(bincenters)])

    transitions = []

    for trj in trjs_discrete:
        
        transitions_trj = []

        #get initial state
        last_ensemble = macrostates_discrete[trj[0]]

        for i in range(len(trj)-lag_time):
    
            current_macrostate = macrostates_discrete[trj[i+1]]

            if current_macrostate == -1:
                current_ensemble = last_ensemble
            else:
                current_ensemble = current_macrostate

            transitions.append([trj[i+1]*nm + current_ensemble, trj[i]*nm + last_ensemble])

            last_ensemble = current_ensemble

        transitions += transitions_trj


    tpm, states_in_order = MSM_methods.transitions_2_msm(transitions)
    if show_TPM:
        plt.matshow(tpm)
        plt.show()

    eqp_msm = MSM_methods.tpm_2_eqprobs(tpm)

    #generalize
    p_msm_a = []
    x_msm_a = []
    p_msm_b = []
    x_msm_b = []

    p_msm_total = []

    for i, so in enumerate(states_in_order):
        if so%2 == 0:
            x_msm_a.append(bincenters[int(so//2)])
            p_msm_a.append(eqp_msm[i])
        else:
            x_msm_b.append(bincenters[int(so//2)])
            p_msm_b.append(eqp_msm[i])

    #x_msm = [bincenters[i] for i in states_in_order]

    plt.plot(x_msm_a, p_msm_a)
    plt.plot(x_msm_b, p_msm_b)


    #assemble halves of the energy landscape
    ha_sio_config = []
    ha_eqp_config = []
    
    for i in range(0, len(bincenters)*2, 2):
        
        config_state_prob = 0
        
        if i in states_in_order:
            config_state_prob += eqp_msm[states_in_order.index(i)][0]
        if i+1 in states_in_order:
            config_state_prob += eqp_msm[states_in_order.index(i+1)][0]
        
        #if len(config_state_prob) > 1:
        #    return "error; too many energies"
        
        ha_sio_config.append(bincenters[int(i/2)])
        ha_eqp_config.append(config_state_prob)

    plt.plot(ha_sio_config, ha_eqp_config)

    mfpts = []

    #for i in range(nm):
    

    mfpt_ab = np.dot(ha_eqp_config, tpm[i])
