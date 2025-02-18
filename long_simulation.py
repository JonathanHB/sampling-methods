import numpy as np
import matplotlib.pyplot as plt
import MSM_methods

def run_long_parallel_simulations(propagator, system, kT, x_init_coord, dt, nsteps, save_period, n_parallel):
    
    #---------------------------------------------------------------
    #run n_parallel long simulations
    
    #set initial positions
    x_init = np.array([x_init_coord for element in range(n_parallel)])
    long_trjs = np.array(propagator(system, kT, x_init, dt, nsteps, save_period))

    #x_init is modified by being fed in to the propagator
    x_init_2 = np.array([x_init_coord for element in range(n_parallel)])

    #include the starting frame
    return np.concatenate((x_init_2.reshape(1,n_parallel), long_trjs)) 



def estimate_eq_pops_histogram(trjs, system, nbins):
    
    #flatten trajectory since the order of the frames does not matter here
    trj_flat = trjs.flatten()

    #-------define bins--------------------------------------------------------------------

    binbounds, bincenters, step = system.analysis_bins(nbins)

    #-------bin trajectory--------------------------------------------------------------------

    #note that digitize reserves the index 0 for entries below the first bin, 
    # but in this case the bins have been constructed so that all entries lie within the explicit bin range
    # so the first and last bins of eq_pops should be empty
    binned_trj = np.digitize(trj_flat, bins = binbounds)

    eq_pops = np.zeros(nbins+2)
    for b in binned_trj:
        eq_pops[b] += 1/len(trj_flat)

    return bincenters, eq_pops



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

    #in steps
    mfpts = save_period*np.reciprocal(n_transitions)*frames_by_state
    
    return n_transitions, mfpts



def msm_analysis(trjs, nbins, system, save_period, lag_time=1, show_TPM=False):

    #get bin boundaries
    trj_flat = trjs.flatten()
    binbounds, bincenters, step = system.analysis_bins(nbins)

    #-------build MSM--------------------------------------------------------

    trj_discrete = np.digitize(trjs, bins = binbounds)
    transitions = [[trj_discrete[i][j], trj_discrete[i+lag_time][j]] for j in range(trj_discrete.shape[1]) for i in range(len(trj_discrete)-lag_time) ]

    tpm, states_in_order = MSM_methods.transitions_2_msm(transitions)
    if show_TPM:
        plt.matshow(tpm)
        plt.show()

    eqp_msm_init = MSM_methods.tpm_2_eqprobs(tpm)
    x_msm = [bincenters[i] for i in states_in_order]

    #--------calculate MFPTS via steady state flux into an artificial sink macrostate----------------------------------------------------------------
    # this approach is going to be slightly wrong since it achieves only an approximate steady state, but it the steady state at least appears to be quite stable.
    # history augmented MSMs should be used instead anyway. See below for the implementation

    mfpts = MSM_methods.calc_MFPT(tpm, x_msm, eqp_msm_init, system.macrostate_classifier, system.n_macrostates, lag_time, save_period)

    return x_msm, eqp_msm_init, mfpts



#get a list of history augmented transitions from a list of parallel trajectories
def __get_ha_transitions(trjs_discrete, macrostates_discrete, n_macrostates, lag_time):

    transitions = []

    for trj in trjs_discrete:
        
        transitions_trj = []

        #get initial state
        last_ensemble = macrostates_discrete[trj[0]]
        
        for i in range(len(trj)-lag_time):
    
            current_macrostate = macrostates_discrete[trj[i+lag_time]]

            #if a macrostate is reached, that is now the ensemble. If in the no man's land, remain in the ensemble of the last macrostate
            if current_macrostate == -1:
                current_ensemble = last_ensemble
            else:
                current_ensemble = current_macrostate

            #record transition
            transitions.append([trj[i]*n_macrostates + last_ensemble, trj[i+lag_time]*n_macrostates + current_ensemble])

            #update buffer
            last_ensemble = current_ensemble

        transitions += transitions_trj

    return transitions



def get_ha_transitions(trjs, nbins, system, lag_time=1):

    #get bin boundaries
    binbounds, bincenters, step = system.analysis_bins(nbins)

    #bin trajectories in configurational space and assign the bins to macrostates
    trjs_discrete = np.digitize(trjs, bins = binbounds).transpose()
    macrostates_discrete = [system.macro_class(x) for x in bincenters]

    #get a list of all the transitions
    ha_transitions = __get_ha_transitions(trjs_discrete, macrostates_discrete, system.n_macrostates, lag_time)

    return ha_transitions


# #put this into analysis methods and have it take the transitions as an argument
# def hamsm_analysis(trjs, nbins, system, save_period, lag_time=1, show_TPM=False):

#     #for consiceness
#     nm = system.n_macrostates

#     #get bin boundaries
#     binbounds, bincenters, step = system.analysis_bins(nbins)

#     #bin trajectories in configurational space and assign the bins to macrostates
#     trjs_discrete = np.digitize(trjs, bins = binbounds).transpose()
#     macrostates_discrete = [system.macro_class(x) for x in bincenters]

#     #get a list of all the transitions
#     ha_transitions = get_ha_transitions(trjs_discrete, macrostates_discrete, nm, lag_time)


#     #-----------------------------------------------------------------------------------------------------------------
#     #build MSM
#     tpm, states_in_order = MSM_methods.transitions_2_msm(ha_transitions)
#     if show_TPM:
#         plt.matshow(tpm)
#         plt.show()

#     eqp_msm = MSM_methods.tpm_2_eqprobs(tpm)


#     #-----------------------------------------------------------------------------------------------------------------
#     #get populations in configuration space (along x) for each ensemble

#     x_ensembles = [[] for element in range(nm)]
#     p_ensembles = [[] for element in range(nm)]

#     for i, so in enumerate(states_in_order):
#         for j in range(nm):
#             if so%nm == j:
#                 x_ensembles[j].append(bincenters[int(so//nm)])
#                 p_ensembles[j].append(eqp_msm[i][0])


#     #-----------------------------------------------------------------------------------------------------------------
#     #assemble halves of the energy landscape to get the overall energy

#     ha_sio_config = []
#     ha_eqp_config = []
    
#     for i in range(0, len(bincenters)*2, 2):
        
#         ha_sio_config.append(bincenters[int(i/2)])
#         ha_eqp_config.append(sum([eqp_msm[states_in_order.index(i+j)][0] if i+j in states_in_order else 0 for j in range(nm)]))


#     #-----------------------------------------------------------------------------------------------------------------
#     #calculate mfpts
#     mfpts = MSM_methods.calc_ha_mfpts(states_in_order, eqp_msm, tpm, macrostates_discrete, nm, save_period)


#     return ha_sio_config, ha_eqp_config, x_ensembles, p_ensembles, mfpts
