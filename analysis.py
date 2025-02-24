#analysis.py
#Jonathan Borowsky
#2/21/25

#functions for comparing the results of different methods 
# of estimating energy lanscapes and kinetics 
# to the true landscapes and to each other

################################################################################################################

import numpy as np
import MSM_methods
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------------------------
#compare true and estimated populations for a single estimation method

#parameters
# system: the system object
# kT: the temperature of the system
# coordinates: the x-coordinates at which equilibrium populations have been estimated
# eq_pops_simulation: the estimated equilibrium populations
# metrics: a list of metrics to compute`
# ensemble_data: a list of tuples of (x-coordinates, populations) for each ensemble of a history augmented MSM

#returns
# returns: a dictionary of metrics computed`
def landscape_comparison(system, kT, coordinates, eq_pops_simulation, metrics = [], ensemble_data = []):

    #compute true populations 
    eq_pops_analytic, energies_analytic = system.normalized_pops_energies(kT, coordinates)

    #unnecessary extra precision; requires bin boundaries as an argument
    #bincenters, eq_pops_analytic = system.compute_true_populations(bins, kT, tolerance = 0.01)
    #plt.plot(bincenters, eq_pops_analytic)
    
    plt.plot(coordinates, eq_pops_analytic, linestyle="dashed")
    plt.plot(coordinates, eq_pops_simulation)

    #plot energy landscapes for each ensemble of a history augmented MSM if provided
    if ensemble_data:
        for xei, xpi in zip(ensemble_data[0], ensemble_data[1]):
            plt.plot(xei, xpi)
    
    plt.show()

    mean_binwidth = (coordinates[-1]-coordinates[0])/(len(coordinates)-1)

    metrics_out = {}

    #note that these metrics currently do not care how badly mis-localized any misplaced probability density is; 
    # there's probably a KDE approach that would do better with that
    if "rmsew" in metrics:
        #without weighting, extending the region over which the RMS error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        rmse_weighted = np.sqrt(np.mean([epa*(eps-epa)**2 for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])/mean_binwidth**3)
        print(f"weighted RMSE = {rmse_weighted}")
        metrics_out["rmsew"] = rmse_weighted

    if "maew" in metrics:
        #without weighting, extending the region over which the mean absolute error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        mae_weighted = np.mean([epa*abs(eps-epa) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])/mean_binwidth**2
        print(f"weighted MAE = {mae_weighted}")            
        metrics_out["maew"] = mae_weighted

    return metrics_out
        
    #kl divergence has bad numerical properties when any estimated entries are 0, which some usually are
    #kl_divergence = sum([epa*np.log(epa/eps) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])
    #print(f"kl divergence = {kl_divergence}")


#----------------------------------------------------------------------------------------------------------------
#helper function for printing mean first passage times neatly from matrices thereof

#parameters
# mfpts: a matrix of mean first passage times
# digits: the number of decimal places to print

#returns
# returns: None
def print_mfpts_2states(mfpts, digits = 0):
    inter_well_mpfts = [mfpts[0,1], mfpts[1,0]]

    meanfmt = f"{np.mean(inter_well_mpfts):.{digits}f}"
    stdfmt = f"{np.std(inter_well_mpfts):.{digits}f}"
    print(f"MFPT = {meanfmt}+-{stdfmt} steps")


#----------------------------------------------------------------------------------------------------------------
#construct a history augmented markov state model from a set of transitions

#parameters
# ha_transitions: a list of transitions, each of which is a tuple of (from_state, to_state)
# nbins: the number of bins to use for the analysis
# system: the system object
# save_period: the number of time steps between frames 
#   (assumed to be equal to the lag time since haMSMs do not require 
#   a lag time sufficient for their microstates to be markovian)
# show_TPM: whether to show the transition probability matrix

#returns
# ha_x_config: the x-coordinates of the bins for which populations were computed (same as bincenters??)
# ha_eqp_config: the equilibrium populations of the configurational bins derived by summing over all ensembles
# x_ensembles: a list of lists of x-coordinates for each ensemble for which state populations were computed
# p_ensembles: a list of lists of equilibrium populations for each ensemble for which state populations were computed
# mfpts: a matrix of mean first passage times between the macrostates
#   (the first index is the from_state and the second index is the to_state)

def hamsm_analysis(ha_transitions, nbins, system, save_period, show_TPM=False):

    #for consiceness
    nm = system.n_macrostates

    #get bin boundaries
    binbounds, bincenters, step = system.analysis_bins(nbins)

    #assign the bins to macrostates
    macrostates_discrete = [system.macro_class(x) for x in bincenters]

    #-----------------------------------------------------------------------------------------------------------------
    #build MSM
    tpm, states_in_order = MSM_methods.transitions_2_msm(ha_transitions)
    if show_TPM:
        plt.matshow(tpm)
        plt.show()

    eqp_msm = MSM_methods.tpm_2_eqprobs(tpm)


    #-----------------------------------------------------------------------------------------------------------------
    #get populations in configuration space (along x) for each ensemble

    x_ensembles = [[] for element in range(nm)]
    p_ensembles = [[] for element in range(nm)]

    for i, so in enumerate(states_in_order):
        for j in range(nm):
            if so%nm == j:
                x_ensembles[j].append(bincenters[int(so//nm)])
                p_ensembles[j].append(eqp_msm[i][0])


    #-----------------------------------------------------------------------------------------------------------------
    #assemble halves of the energy landscape to get the overall energy

    ha_x_config = []
    ha_eqp_config = []
    
    for i in range(0, len(bincenters)*2, 2):
        
        ha_x_config.append(bincenters[int(i/2)]) # I think it's fine to just return bincenters
        ha_eqp_config.append(sum([eqp_msm[states_in_order.index(i+j)][0] if i+j in states_in_order else 0 for j in range(nm)]))


    #-----------------------------------------------------------------------------------------------------------------
    #calculate mfpts
    mfpts = MSM_methods.calc_ha_mfpts(states_in_order, eqp_msm, tpm, macrostates_discrete, nm, save_period)


    return ha_x_config, ha_eqp_config, x_ensembles, p_ensembles, mfpts


#-----------------------
#bootstrapping

#parameters
# n_bootstrap: number of times to run each method

#returns


def bootstrap_method_comparison(n_bootstrap, analysis_methods, system, kT, dt, propagator, nsteps, save_period, n_parallel):
    
    mfpts = []
    populations = []

    for m in analysis_methods:
        print(m)

        method_mfpts = []
        method_coords = []
        method_probabilities = []

        for bi in range(n_bootstrap):
            print(f"round {bi}")

            coords, probs, mfpts = m(system, kT, dt, propagator, nsteps, save_period, n_parallel)
            method_mfpts.append(mfpts)
            method_coords.append(coords)
            method_probabilities.append(probs)

        mfpts.append([np.mean(method_mfpts), np.std(method_mfpts)])
        
        #calculate energy landscape with error bars, accounting for the fact that not all landscape estimators will yield estimates for the same states due to ergodic trimming of MSMs
        method_coords_all = np.unique(method_mfpts)
        method_coords_all = np.sort(method_coords_all)
        for mc in method_coords_all






        lag_time = 1

        long_trjs = long_simulation.run_long_parallel_simulations(propagator, system, kT, system.standard_init_coord, dt, nsteps, save_period, n_parallel)
        print(f"simulation steps:\n Aggregate: {nsteps*n_parallel} \n Molecular: {nsteps}")


        #------------------------------------------------------------------------------------------
        #non-MSM analysis
        x, p = long_simulation.estimate_eq_pops_histogram(long_trjs, system1, nbins)
        transitions, mfpts = long_simulation.calc_mfpt(system1.macro_class, system1.n_macrostates, save_period, long_trjs)
        if n_bootstrap == 1:
            metrics = analysis.landscape_comparison(system1, kT, x, p, metrics = ["maew"])
            analysis.print_mfpts_2states(mfpts)

        inter_well_mpfts = [mfpts[0,1], mfpts[1,0]]
        mfpts_long_raw.append(np.mean(inter_well_mpfts))
