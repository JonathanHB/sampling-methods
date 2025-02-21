#long_simulations.py
#Jonathan Borowsky
#2/21/25

#functions for running and analyzing long simulations
# to obtain unbiased estimates of equilibrium populations and kinetics 
# to compare adaptive sampling methods against

################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import MSM_methods


#---------------------------------------------------------------------------------
#run a set of parallel simulations

#parameters:
# propagator: integrator that generates the trajectory
# system: system object that contains the potential energy and force functions
# kT: temperature at which to simulate
# x_init_coord: initial coordinates for the system
# dt: time step for the simulation
# nsteps: number of steps for which to run the simulation
# save_period: how often to save trajectory frames
# n_parallel: number of parallel simulations to run

# returns: a set of trajectories, one for each parallel simulation

def run_long_parallel_simulations(propagator, system, kT, x_init_coord, dt, nsteps, save_period, n_parallel):
    
    #set initial positions
    x_init = np.array([x_init_coord for element in range(n_parallel)])
    long_trjs = np.array(propagator(system, kT, x_init, dt, nsteps, save_period))

    #x_init is modified by being fed in to the propagator
    x_init_2 = np.array([x_init_coord for element in range(n_parallel)])

    #include the starting frame
    return np.concatenate((x_init_2.reshape(1,n_parallel), long_trjs)) 


#---------------------------------------------------------------------------------
#estimate the equilibrium populations from a set of parallel trajectories by counting up where the system spends its time
#parameters:
# trjs: a set of parallel trajectories
# system: system object that contains the potential energy and force functions
# nbins: number of bins to use for the histogram

# returns: 
# bincenters: the centers of the bins for which the equilibrium populations are calculated
# eq_pops: the equilibrium populations for each bin
# note that the first and last bins extend to -inf and inf respectively, 
#   but the bin centers are placed as if they were spaced equally to the other bins

def estimate_eq_pops_histogram(trjs, system, nbins):
    
    #flatten trajectory since the order of the frames does not matter here
    trj_flat = trjs.flatten()

    # define bins
    binbounds, bincenters, step = system.analysis_bins(nbins)

    #bin trajectory

    #note that digitize reserves the index 0 for entries below the first bin, 
    # but in this case the bins have been constructed so that all entries lie within the explicit bin range
    # so the first and last bins of eq_pops should be empty
    binned_trj = np.digitize(trj_flat, bins = binbounds)

    eq_pops = np.zeros(nbins+2)
    for b in binned_trj:
        eq_pops[b] += 1/len(trj_flat)

    return bincenters, eq_pops


#---------------------------------------------------------------------------------
#estimate the mean first passage time between macrostates from a set of parallel trajectories
#  by counting the macrostate transitions

#parameters:
# macrostate_classifier: function that takes in a coordinate and returns the macrostate index
#   or -1 if the coordinate is in no macrostate
#   note that the non-macrostate region is not usually equivalent to an additional macrostate because 
#   transitions between macrostates should be much slower than those within them
#   and the non-macrostate region is usually a high energy region with a convex energy profile for which this is not true
# n_macrostates: number of macrostates
# save_period: the number of steps between trajectory frames
# trajectories: a list of parallel trajectories

#returns:
# n_transitions: a matrix of the number of transitions between macrostates
# mfpts: a matrix of the mean first passage times between macrostates, in units of integrator steps
#   the mfpt from macrostate i to j is given by mfpts[j][i]; same indexing for transition counts

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


#---------------------------------------------------------------------------------
#DEPRECATED
#regular MSM analysis, should be broken into a get_transitions() method here and an msm_anslysis() method in analysis.py
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


#---------------------------------------------------------------------------------
#get a list of history augmented transitions from a list of parallel trajectories

#parameters:
# trjs_discrete: a list of parallel trajectories, each trajectory is a list of the bin index of the trajectory frame
# macrostates_discrete: a list of the macrostate index for each bin center
# n_macrostates: the number of macrostates
# lag_time: the number of frames over which to look for transitions
#    note that overlapping segments of n_frames are used, 
#    so the transitions are not independent at lag times greater than 1
#    I do not know if it is a problem if this lag time differs from that used to build the haMSM

#returns:
# transitions: a list of transitions, each transition is a list of the form [start state, end state]
#   where the start and end states are given by the configurational bin index of the trajectory frame 
#   multiplied by the number of macrostates, plus the ensemble index
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


#---------------------------------------------------------------------------------
# wrapper function for __get_ha_transitions()
# get a list of history augmented transitions from a list of parallel trajectories

#parameters:
# trjs: a list of parallel trajectories
# nbins: the number of configurational bins to use
# system: system object that contains the macrostate classifier and information on the number of macrostates
# lag_time: the number of frames over which to look for transitions
#    note that overlapping segments of n_frames are used, 
#    so the transitions are not independent at lag times greater than 1
#    I do not know if it is a problem if this lag time differs from that used to build the haMSM

#returns:
# ha_transitions: a list of transitions, each transition is a list of the form [start state, end state]
#   where the start and end states are given by the configurational bin index of the trajectory frame 
#   multiplied by the number of macrostates, plus the ensemble index
def get_ha_transitions(trjs, nbins, system, lag_time=1):

    #get bin boundaries
    binbounds, bincenters, step = system.analysis_bins(nbins)

    #bin trajectories in configurational space and assign the bins to macrostates
    trjs_discrete = np.digitize(trjs, bins = binbounds).transpose()
    macrostates_discrete = [system.macro_class(x) for x in bincenters]

    #get a list of all the transitions
    ha_transitions = __get_ha_transitions(trjs_discrete, macrostates_discrete, system.n_macrostates, lag_time)

    return ha_transitions

