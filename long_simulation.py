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
import analysis
import propagators
import metadynamics
#import deeptime


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

def run_long_parallel_simulations(propagator, system, kT, dt, nsteps, save_period, n_parallel):
    
    #set initial positions
    if system.start_from_index:
        sim_init_coord = system.standard_init_index
    else:
        sim_init_coord = system.standard_init_coord

    x_init = np.array([sim_init_coord for element in range(n_parallel)])
    
    #propagate
    long_trjs = np.array(propagator(system, kT, x_init, dt, nsteps, save_period)).transpose(1,0,2)

    #x_init is modified by being fed in to the propagator
    x_init_2 = np.array([sim_init_coord for element in range(n_parallel)]).reshape((n_parallel, 1, len(system.standard_init_coord)))

    #include the starting frame
    #'1' here is not a magic number; it instead reflects the fact that the starting coordinate is a single frame long
    return np.concatenate((x_init_2, long_trjs), axis=1) 


def resume_parallel_simulations(propagator, system, kT, dt, nsteps, save_period, prev_trj):

    x_init = np.array([xii[-1] for xii in prev_trj])
    
    #propagate
    long_trjs = np.array(propagator(system, kT, x_init, dt, nsteps, save_period)).transpose(1,0,2)

    #x_init is modified by being fed in to the propagator
    #x_init_2 = np.array([xii[-1] for xii in prev_trj]).reshape((len(prev_trj), 1, len(system.standard_init_coord)))

    #include the starting frame
    #'1' here is not a magic number; it instead reflects the fact that the starting coordinate is a single frame long
    return np.concatenate((prev_trj, long_trjs), axis=1) 


def resume_parallel_simulations_mtd(propagator, system, kT, dt, nsteps, save_period, prev_trj, grid):

    x_init = np.array([xii[-1] for xii in prev_trj])
    
    #propagate
    long_trjs, grid = propagator(system, kT, x_init, dt, nsteps, save_period, grid)
    long_trjs = np.array(long_trjs).transpose(1,0,2)

    #x_init is modified by being fed in to the propagator
    #x_init_2 = np.array([xii[-1] for xii in prev_trj]).reshape((len(prev_trj), 1, len(system.standard_init_coord)))

    #include the starting frame
    #'1' here is not a magic number; it instead reflects the fact that the starting coordinate is a single frame long
    return np.concatenate((prev_trj, long_trjs), axis=1), grid
#---------------------------------------------------------------------------------
#estimate the equilibrium populations from a set of parallel trajectories by counting up where the system spends its time
# parameters:
# trjs: a set of parallel trajectories
# system: system object that contains the potential energy and force functions
# nbins: number of bins to use for the histogram

# returns: 
# bincenters: the centers of the bins for which the equilibrium populations are calculated
# eq_pops: the equilibrium populations for each bin
# note that the first and last bins extend to -inf and inf respectively, 
#   but the bin centers are placed as if they were spaced equally to the other bins

def estimate_eq_pops_histogram(trjs, system, nbins):

    bincenters_flat, binwidth, nbins, actual_nbins, binbounds, ndim, prods_higher = analysis.construct_voxel_bins(system.standard_analysis_range, nbins)
    binned_trj = analysis.bin_to_voxels(ndim, binbounds, prods_higher, trjs)
    #binned_trj, bincenters, binwidth, actual_nbins, binbounds = analysis.digitize_to_voxel_bins(system.standard_analysis_range, nbins, trjs)
    binned_trj = np.array(binned_trj).flatten()

    #flatten trajectory since the order of the frames does not matter here
    trj_flat = trjs.flatten()

    # define bins
    #binbounds, bincenters, step = system.analysis_bins_1d(nbins)
    #actual_nbins = nbins
    #bin trajectory

    #note that digitize reserves the index 0 for entries below the first bin, 
    # but in this case the bins have been constructed so that all entries lie within the explicit bin range
    # so the first and last bins of eq_pops should be empty
    #binned_trj = np.digitize(trj_flat, bins = binbounds)

    eq_pops = np.zeros(actual_nbins)
    for b in binned_trj:
        eq_pops[b] += 1/len(trj_flat)

    bins_sampled = np.sort(np.unique(binned_trj))
    bincenters_sampled = [bincenters_flat[i] for i in bins_sampled]
    eqp_sampled = [eq_pops[i] for i in bins_sampled]

    return bincenters_flat, eq_pops, bincenters_sampled, eqp_sampled


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

    #in steps
    mfpts = save_period*np.reciprocal(n_transitions)*frames_by_state
    
    return n_transitions, mfpts


#---------------------------------------------------------------------------------
def __get_transitions(trjs_discrete, lag_time):

    transitions = []

    for trj in trjs_discrete:
        
        transitions_trj = []
        #print(trj[0])

        #get initial state
        #last_ensemble = macrostates_discrete[trj[0]]
        #print(trj.shape)
        for i in range(len(trj)-lag_time):
    
            #current_macrostate = macrostates_discrete[trj[i+lag_time]]

            #if a macrostate is reached, that is now the ensemble. If in the no man's land, remain in the ensemble of the last macrostate
            # if current_macrostate == -1:
            #     current_ensemble = last_ensemble
            # else:
            #     current_ensemble = current_macrostate

            #record transition
            transitions.append([trj[i], trj[i+lag_time]])

            #update buffer
            #last_ensemble = current_ensemble

        transitions += transitions_trj

    return transitions


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
        #print(trj[0])

        #get initial state
        last_ensemble = macrostates_discrete[trj[0]]
        #print(trj.shape)
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

    #analysis.digitize_to_voxel_bins(system.standard_analysis_range, nbins, trjs)

    bincenters_flat, binwidth, nbins, actual_nbins, binbounds, ndim, prods_higher = analysis.construct_voxel_bins(system.standard_analysis_range, nbins)
    trjs_discrete = analysis.bin_to_voxels(ndim, binbounds, prods_higher, trjs)
    #, bincenters, binwidth, actual_nbins, binbounds
    #digitize_to_voxel_bins(system.standard_analysis_range, nbins, trjs)
    # print(len(trjs_discrete))
    # print(len(trjs_discrete[0]))
    # print(np.stack(trjs_discrete).shape)
    
    #get bin boundaries
    #binbounds, bincenters, step = system.analysis_bins_1d(nbins)

    #bin trajectories in configurational space and assign the bins to macrostates
    #trjs_discrete = np.digitize(trjs, bins = binbounds).reshape((trjs.shape[0], trjs.shape[1]))
    #print(trjs_discrete.shape)
    #print("get_ha_transitions")
    #print([x for x in bincenters])
    macrostates_discrete = [system.macro_class(x) for x in bincenters_flat]

    #get a list of all the transitions
    ha_transitions = __get_ha_transitions(np.stack(trjs_discrete), macrostates_discrete, system.n_macrostates, lag_time)

    return ha_transitions


#---------------------------------------------------------------------------------
#run a long simulation and return the results of brute force (histogram) analysis for bootstrapping

#parameters:
# TODO bundle stuff like the system, timestep, and propagator 
#   (and maybe the temperature for now, but it would be nice to try 
#   varying it later in some enhanced simulation methods)
# system: system object that contains the potential energy and force functions
# kT: temperature at which to simulate
# dt: time step for the simulation
# propagator: integrator that generates the trajectory
# aggregate_simulation_limit: the maximum total number of steps to run 
#   (TODO later on experiment with further limits on molecular time)

# returns: 
# coordinates
# equilibrium populations
# mean first passage time matrix
#    #consider returning transition counts as well to allow for comparison of the number of independent transitions, 
#    which at least in some cases is probably a good proxy for sampling quality
# aggregate number of simulation steps


def long_simulation_histogram_analysis(system, kT, dt, aggregate_simulation_limit, n_parallel, save_period, n_analysis_bins, n_timepoints):
    
    #run simulation
    nsteps = int(round(aggregate_simulation_limit/(n_parallel*n_timepoints)))

    #set initial positions
    if system.start_from_index:
        sim_init_coord = system.standard_init_index
    else:
        sim_init_coord = system.standard_init_coord

    long_trjs = np.array([sim_init_coord for element in range(n_parallel)]).reshape((n_parallel, 1, len(system.standard_init_coord)))

    agg_times = []
    xs_t = []
    ps_t = []
    mfpts_t = []

    for tp in range(n_timepoints):
        long_trjs = resume_parallel_simulations(propagators.propagate, system, kT, dt, nsteps, save_period, long_trjs)

        #analysis
        x, p, xs, ps = estimate_eq_pops_histogram(long_trjs, system, n_analysis_bins)
        transitions, mfpts = calc_mfpt(system.macro_class, system.n_macrostates, save_period, long_trjs)

        agg_times.append(nsteps*n_parallel*(tp+1))
        xs_t.append(xs)
        ps_t.append(ps)
        mfpts_t.append(mfpts)

    return agg_times, xs_t, ps_t, mfpts_t


#------------------------------------------------------------------------------------
# METADYNAMICS simulation
#------------------------------------------------------------------------------------

def long_simulation_histogram_analysis_mtd(system, kT, dt, aggregate_simulation_limit, n_parallel, save_period, n_analysis_bins, n_timepoints):
    
    #run simulation
    nsteps = int(round(aggregate_simulation_limit/(n_parallel*n_timepoints)))

    #set initial positions
    if system.start_from_index:
        sim_init_coord = system.standard_init_index
    else:
        sim_init_coord = system.standard_init_coord

    long_trjs = np.array([sim_init_coord for element in range(n_parallel)]).reshape((n_parallel, 1, len(system.standard_init_coord)))

    grid = metadynamics.grid(system.standard_analysis_range, n_analysis_bins, rate = 0.1)

    agg_times = []
    xs_t = []
    ps_t = []
    mfpts_t = []

    for tp in range(n_timepoints):
        long_trjs, grid = resume_parallel_simulations_mtd(propagators.propagate_mtd, system, kT, dt, nsteps, save_period, long_trjs, grid)

        #analysis
        x, p, xs, ps = estimate_eq_pops_histogram(long_trjs, system, n_analysis_bins)
        transitions, mfpts = calc_mfpt(system.macro_class, system.n_macrostates, save_period, long_trjs)

        agg_times.append(nsteps*n_parallel*(tp+1))
        xs_t.append(xs)
        ps_t.append(ps)
        mfpts_t.append(mfpts)

        plt.plot(grid.bincenters, grid.grid)

    plt.show()
    return agg_times, xs_t, ps_t, mfpts_t


#---------------------------------------------------------------------------------
#run a long simulation and return the results of haMSM analysis for bootstrapping

#parameters:
# TODO bundle stuff like the system, timestep, and propagator 
#   (and maybe the temperature for now, but it would be nice to try 
#   varying it later in some enhanced simulation methods)
# system: system object that contains the potential energy and force functions
# kT: temperature at which to simulate
# dt: time step for the simulation
# propagator: integrator that generates the trajectory
# aggregate_simulation_limit: the maximum total number of steps to run 
#   (TODO later on experiment with further limits on molecular time)

# returns: 
# coordinates
# equilibrium populations
# mean first passage times between two macrostates
#    TODO generalize to n macrostates; returning an upper triangular matrix or something (or just the whole square matrix)
#    TODO this will also permit inspection of self transitions
#    #consider returning transition counts as well to allow for comparison of the number of independent transitions, 
#    which at least in some cases is probably a good proxy for sampling quality

def long_simulation_hamsm_analysis(system, kT, dt, aggregate_simulation_limit, n_parallel, save_period, n_analysis_bins):

    #run simulation
    nsteps = int(round(aggregate_simulation_limit/n_parallel))
    long_trjs = run_long_parallel_simulations(propagators.propagate, system, kT, dt, nsteps, save_period, n_parallel)

    #analysis
    #note that lag time is measured in saved frames
    ha_transitions = get_ha_transitions(long_trjs, n_analysis_bins, system, lag_time=1)
    x, p, xs, ps, x_ens, p_ens, mfpts = analysis.hamsm_analysis(ha_transitions, n_analysis_bins, system, save_period, lag_time=1, show_TPM=False)

    return nsteps*n_parallel, xs, ps, mfpts



#---------------------------------------------------------------------------------


def long_simulation_msm_analysis(system, kT, dt, aggregate_simulation_limit, n_parallel, save_period, n_analysis_bins, n_timepoints):

    bincenters_flat, binwidth, nbins, actual_nbins, binbounds, ndim, prods_higher = analysis.construct_voxel_bins(system.standard_analysis_range, n_analysis_bins)

    #run simulation
    nsteps = int(round(aggregate_simulation_limit/(n_parallel*n_timepoints)))

    #set initial positions
    if system.start_from_index:
        sim_init_coord = system.standard_init_index
    else:
        sim_init_coord = system.standard_init_coord

    long_trjs = np.array([sim_init_coord for element in range(n_parallel)]).reshape((n_parallel, 1, len(system.standard_init_coord)))

    agg_times = []
    xs_t = []
    ps_t = []
    mfpts_t = []

    for tp in range(n_timepoints):
        long_trjs = resume_parallel_simulations(propagators.propagate, system, kT, dt, nsteps, save_period, long_trjs)

        #analysis
        #note that lag time is measured in saved frames
        #ha_transitions = get_ha_transitions(long_trjs, n_analysis_bins, system, lag_time=1)
        
        trjs_discrete = analysis.bin_to_voxels(ndim, binbounds, prods_higher, long_trjs)
        #trjs_discrete, bincenters, binwidth, actual_nbins, binbounds = analysis.digitize_to_voxel_bins(system.standard_analysis_range, n_analysis_bins, long_trjs)

        transitions=__get_transitions(np.stack(trjs_discrete), lag_time=1)

        #x, p, xs, ps, x_ens, p_ens, mfpts = analysis.hamsm_analysis(ha_transitions, n_analysis_bins, system, save_period, lag_time=1, show_TPM=False)
        xs, ps = MSM_methods.transitions_to_eq_probs(transitions, bincenters_flat, show_TPM=False)

        agg_times.append(nsteps*n_parallel*(tp+1))
        xs_t.append(xs)
        ps_t.append(ps)
        mfpts_t.append([])

    return agg_times, xs_t, ps_t, mfpts_t


#    return nsteps*n_parallel, xs, ps, []


# def long_simulation_msm_analysis(system, kT, dt, aggregate_simulation_limit, n_parallel, save_period, n_analysis_bins):

#     #run simulation
#     nsteps = int(round(aggregate_simulation_limit/n_parallel))
#     long_trjs = run_long_parallel_simulations(propagators.propagate, system, kT, dt, nsteps, save_period, n_parallel)

#     #analysis
#     #note that lag time is measured in saved frames
#     #ha_transitions = get_ha_transitions(long_trjs, n_analysis_bins, system, lag_time=1)
    
#     trjs_discrete, bincenters, binwidth, actual_nbins, binbounds = analysis.digitize_to_voxel_bins(system.standard_analysis_range, n_analysis_bins, long_trjs)

#     transitions=__get_transitions(np.stack(trjs_discrete), lag_time=1)

#     #x, p, xs, ps, x_ens, p_ens, mfpts = analysis.hamsm_analysis(ha_transitions, n_analysis_bins, system, save_period, lag_time=1, show_TPM=False)
#     xs, ps = MSM_methods.transitions_to_eq_probs(transitions, bincenters, show_TPM=False)


#     return nsteps*n_parallel, xs, ps, []