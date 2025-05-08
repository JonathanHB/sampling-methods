#Method for running weighted ensemble

#IDEA: can populate bins by resuming previously-terminated walkers, especially deeply-branching ones; could include these 'ghost' walkers in RMSD comparisons as well

import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import propagators
import analysis
import MSM_methods
import metadynamics

#parameters
#   x_init: list of floats: coordinates of initial walkers
#   w_init: list of floats: weights of initial walkers (not to be confused with the westpa wynton script of the same name)
#   nrounds: nonnegative int: number of rounds to run
#   nbins: positive int: total number of bins within binrange
#   walkers_per_bin: positive int: number of parallel simulations to initialize per bin
#   binrange: list of 2 floats: progress coordinate range within which to bin simulations
#   propagator: function for running dynamics
#   prop_params: arguments for propagator except x_init
#   macrostate_classifier: a function that takes a walker position and current ensemble and returns an integer denoting a macrostate

#returns
#   [list of floats: coordinates of final walkers
#    list of floats: weights of final walkers,
#    list of floats: bin boundaries,
#    list list of floats: walker x coordinates by round
#    list list of floats: walker weights by round
#    list list of floats: walker ensembles by round
#    list of 2-element lists of floats: all the transitions from the entire WE run
#    ]

#summary
#   for each round:
#      1. bin walkers along x
#      2. for each bin:
#         if number of walkers in bin > walkers_per_bin:
#            weighted randomly merge walkers until number of walkers in bin = walkers_per_bin
#         else if number of walkers in bin < walkers_per_bin:
#            weighted randomly split the original walkers until number of walkers in bin = walkers_per_bin
#      3. propagate dynamics on the resulting x vector
#      4. log data of interest
#      5. pass coordinates, weights, and ensembles to next round 

def weighted_ensemble(x_init, w_init, nrounds, nbins, walkers_per_bin, system, propagator, prop_params, macrostate_classifier, n_macrostates, grid, ha_binning=False):

    split_limit = 2.00001*sys.float_info.min #0.0002
    merge_limit = 1 #effectively no limit
    
    #n_macrostates = 2
    
    #extreme_val = 10**6
    #binbounds = np.concatenate(([-extreme_val], np.linspace(binrange[0], binrange[1], nbins+1), [extreme_val]))
    #if not ha_binning:
    #binbounds = np.linspace(binrange[0], binrange[1], nbins+1)    
    #else:
    #    binbounds = np.linspace(binrange[0], binrange[1], (nbins+1)*n_macrostates)
    
    bincenters_flat, binwidth, nbins, actual_nbins, binbounds, ndim, prods_higher = analysis.construct_voxel_bins(system.standard_analysis_range, nbins)

    #print(len(binbounds))
    
    #data to track across multiple WE runs for analysis
    xtrj = []
    wtrj = []
    etrj = []   
    
    transitions = []
    hamsm_transitions = [] #has twice as many bins to account for different ensembles
    n_trans_by_round = [] #cumulative number of transitions at the end of each round; 
    #for assessment of how many WE rounds are needed for convergence
    #this applies to both transitions and hamsm_transitions
    
    #determine initial walker ensembles
    e_init = [macrostate_classifier(i, -1) for i in x_init]
        
    #add initial coordinates, weights, and ensembles so the trajectory includes both endpoints
    xtrj.append(x_init)
    wtrj.append(w_init) 
    etrj.append(e_init)    

        
    for r in range(nrounds):
        
        if r%round(nrounds/10) == 0:
            print(r)
        #print("-----")
   
        #----------------------------------------------------------------------
        # assign walkers to bins
        
        # bin_inds = the index of the bin to which each value in x_init belongs
        #config_bin_inds = np.digitize(x_init, binbounds)
        #TODO I think this function call is redundant with one of the two similar calls below. A buffer variable might be needed but that would be fine.
        #print(nbins)
        #print(binbounds)
        config_bin_inds, nd_inds = analysis.bin_to_voxels_timeslice(ndim, binbounds, prods_higher, x_init)

        if not ha_binning:
            bin_inds = config_bin_inds
            inds_by_bin = [[] for element in range(actual_nbins)]

        else:
#             print(max(config_bin_inds))
            bin_inds = [cbi*n_macrostates+e_init[einit_ind] for einit_ind, cbi in enumerate(config_bin_inds)]
            inds_by_bin = [[] for element in range((actual_nbins)*n_macrostates)]

        #the indices in x_init of the walkers in each bin
#         print(len(inds_by_bin))
#         print(max(bin_inds))
        for xinit_ind, bin_ind in enumerate(bin_inds):
            #print(bin_ind)
            inds_by_bin[bin_ind].append(xinit_ind)
        

        # if r%round(nrounds/10) == 0:
        #     plt.scatter([bincenters_flat], [len(i) for i in inds_by_bin])
        #     plt.show()
        #     print(f"bins_with_1_walker: {[iii for iii, l in enumerate([len(i) for i in inds_by_bin]) if l == 1]}")
            
        #----------------------------------------------------------------------
        #Merge and/or duplicate walkers
        
        #merge or duplicate walkers in each bin if necessary; 
        # placing the resulting set of walkers into new x and w arrays
        x_md = []
        w_md = []
        e_md = []

        for isi, indset in enumerate(inds_by_bin):
            
            #continue simulations in bins with the right population
            if len(indset) == walkers_per_bin:
                for i in indset:
                    x_md.append(x_init[i])
                    e_md.append(e_init[i])
                    w_md.append(w_init[i])
                
            #duplicate simulations in bins with too few walkers
            elif len(indset) < walkers_per_bin and len(indset) > 0:

                #select walkers to duplicate
                w_indset = [w_init[i] for i in indset]

                duplicated_walkers = random.choices(indset, weights=w_indset, k = walkers_per_bin-len(indset))
                
                #add coordinates and weights of walkers from this bin to the list for next round
                # coordinates are unchanged for duplicated walkers; weights are reduced
                for i in indset:
                    #add multiple copies of walkers to be duplicated with proportionally smaller weights
                    for j in range(1+duplicated_walkers.count(i)):
                        x_md.append(x_init[i])
                        e_md.append(e_init[i])

                        if w_init[i] >= split_limit:
                            #this is the normal WE algorithm
                            w_md.append(w_init[i]/(1+duplicated_walkers.count(i)))
                        else:
                            w_md.append(w_init[i])
                            break #do not duplicate too-light walkers

                            
            #merge simulations in bins with too many walkers
            elif len(indset) > walkers_per_bin:
                #instead of doing what's below just merge the two lightest walkers to prevent probability from accumulating in heavier ones
                
                #total bin weight; does not change because merging operations preserve weight
                w_bin = sum([w_init[i] for i in indset])
            
                #deepcopy; may be unnecessary
                local_indset = [i for i in indset]
                w_local_indset = [w_init[i] for i in indset]

                #remove walkers until only walkers_per_bin remain
                for i in range(len(indset)-walkers_per_bin):
                    
                    #weights for walker elimination from Huber and Kim 1996 appendix A
                    w_removal = [(w_bin - w_init[i])/w_bin for i in local_indset]
                    #pick 1 walker to remove, most likely one with a low weight
                    #the [0] eliminates an unnecessary list layer
                    removed_walker = random.choices([j for j in range(len(local_indset))], weights=w_removal, k = 1)[0]
                    
                    #remove the walker
                    local_indset = [i for ii, i in enumerate(local_indset) if ii != removed_walker ]
                    removed_weight = w_local_indset[removed_walker]
                    w_local_indset = [i for ii, i in enumerate(w_local_indset) if ii != removed_walker ]
                    
                    #pick another walker to gain the removed walker's probability
                    #selection chance is proportional to existing weight
                    recipient_walker = random.choices([j for j in range(len(local_indset))], weights=w_local_indset, k = 1)[0]
                    w_local_indset[recipient_walker] += removed_weight

                for i in range(walkers_per_bin):
                    x_md.append(x_init[local_indset[i]])
                    e_md.append(e_init[local_indset[i]])
                    w_md.append(w_local_indset[i])

        #----------------------------------------------------------------------
        #run dynamics
        x_init = propagator(prop_params[0], prop_params[1], np.array(x_md), prop_params[2], prop_params[3])
        e_init = [macrostate_classifier(i, e) for i, e in zip(x_init, e_md)] #update macrostates based on dynamics
        w_init = w_md #weights are unaffected by dynamics
        
        #----------------------------------------------------------------------        
        #recording
        
        #log walker positions, weights, and ensembles
        xtrj.append(x_init)
        wtrj.append(w_init)
        etrj.append(e_init)
        
        #update grid
        grid.update(x_init, w_init)

        #bin transitions and add them to the transition list
        #TODO This could be made more efficient by separating digitize_voxel_bins into a method to define the bins and another to bin the trajectory.
        #     The former would only need to run once.
        #     we also need a variant that bins a single time slice to fix current bug
        bin_inds_1, nd_inds_1 = analysis.bin_to_voxels_timeslice(ndim, binbounds, prods_higher, x_md)
        bin_inds_2, nd_inds_2 = analysis.bin_to_voxels_timeslice(ndim, binbounds, prods_higher, x_init)

        # bin_inds_1 = np.digitize(x_md, binbounds) #not the same as bin_inds
        # bin_inds_2 = np.digitize(x_init, binbounds)
        
        transitions +=  [[b1,b2] for b1,b2 in zip(bin_inds_1, bin_inds_2)]
        hamsm_transitions += [[b1*n_macrostates+e1,b2*n_macrostates+e2] for b1,b2,e1,e2 in zip(bin_inds_1, bin_inds_2, e_md, e_init)]
        
        n_trans_by_round.append(len(transitions))

        
    return x_init, e_init, w_init, binbounds, xtrj, etrj, wtrj, transitions, hamsm_transitions, n_trans_by_round, bincenters_flat, grid


#wrapper function for weighted_ensemble() that initializes the walkers
def weighted_ensemble_start(x_init_val, nrounds, nbins, walkers_per_bin, system, propagator, prop_params, macrostate_classifier, n_macrostates, ha_binning=False):

    #start 1 bin worth of walkers at x_init_val with equal weights
    x_init = np.array([x_init_val for element in range(walkers_per_bin)])
    w_init = [1/walkers_per_bin for element in range(walkers_per_bin)]
    
    grid = metadynamics.grid(system.standard_analysis_range, nbins)

    #run weighted ensemble with brownian dynamics
    #put this on multiple lines
    return weighted_ensemble(\
                        x_init,\
                        w_init,\
                        nrounds,\
                        nbins,\
                        walkers_per_bin,\
                        system, propagator,\
                        prop_params,\
                        macrostate_classifier,\
                        n_macrostates,\
                        grid,\
                        ha_binning=False
                        )


def weighted_ensemble_hamsm_analysis(system, kT, dt, aggregate_simulation_limit, n_parallel, nsteps, n_analysis_bins):
    
    #N = 500             #total number of walkers within binrange
    #nbins = 40         #total number of bins within binrange 
    #nbins should match the value above, at least for analysis; make a separate n_bins_analysis variable

    walkers_per_bin = int(round(n_parallel/n_analysis_bins))
    print(f"Each bin can hold up to {walkers_per_bin} walkers, for a total of up to about {walkers_per_bin*(n_analysis_bins)} walkers")

    #binrange = system.standard_analysis_range 

    #progress coordinate range within which to bin simulations
                        #this should extend well past the stall point for examination of the WE stall force
                        #the area past either end of binrange is a bin extending to either + or - inf, yielding a total of nbins+2 bins
    n_macrostates=2
            
    #nsteps = save_period        #round length; to match long simulations since MFPT = f(lag time)
    nrounds = int(round(aggregate_simulation_limit/(n_parallel*nsteps)))  #number of WE rounds to run

    x_init_val = system.standard_init_coord

    #run weighted ensemble with brownian dynamics
    #put this on multiple lines
    x_init, e_init, w_init, binbounds, xtrj, etrj, wtrj, transitions, hamsm_transitions, n_trans_by_round, bincenters_flat \
    = weighted_ensemble_start(\
                        x_init_val,\
                        nrounds,\
                        n_analysis_bins,\
                        walkers_per_bin,\
                        system, propagators.propagate_save1,\
                        [system, kT, dt, nsteps],\
                        system.ensemble_class,\
                        n_macrostates,\
                        ha_binning=True)


    aggregate_walkers = len([j for i in xtrj for j in i])
    #print(f"simulation steps:\n Aggregate: {nsteps*aggregate_walkers} \n Molecular: {nsteps*nrounds}")

    #x, p, xs, es = weighted_ensemble_2.landscape_recovery(xtrj, wtrj, binbounds, transitions, hamsm_transitions, n_trans_by_round, nrounds, n_macrostates, system1.potential, system1.macro_class, kT)

    x_hamsm, eqp_hamsm, x_hamsm_sampled, eqp_hamsm_sampled, x_ens, p_ens, mfpts_hamsm = analysis.hamsm_analysis(hamsm_transitions, n_analysis_bins, system, nsteps, lag_time=1, show_TPM=False)

    # if n_bootstrap == 1:
    #     metrics = analysis.landscape_comparison(system1, kT, x_hamsm, eqp_hamsm, metrics = ["maew"], ensemble_data = [x_ens, p_ens])
    #     analysis.print_mfpts_2states(mfpts_hamsm)

    # inter_well_mpfts_we_hamsm = [mfpts_hamsm[0,1], mfpts_hamsm[1,0]]
    # mfpts_we_hamsm.append(np.mean(inter_well_mpfts_we_hamsm))

    return nsteps*aggregate_walkers, x_hamsm_sampled, eqp_hamsm_sampled, mfpts_hamsm



def weighted_ensemble_msm_analysis(system, kT, dt, aggregate_simulation_limit, n_parallel, nsteps, n_analysis_bins, n_timepoints):
    
    #N = 500             #total number of walkers within binrange
    #nbins = 40         #total number of bins within binrange 
    #nbins should match the value above, at least for analysis; make a separate n_bins_analysis variable

    walkers_per_bin = int(round(n_parallel/n_analysis_bins))
    print(f"Each bin can hold up to {walkers_per_bin} walkers, for a total of up to about {walkers_per_bin*(n_analysis_bins)} walkers")

    n_macrostates=1
            
    nrounds = int(round(aggregate_simulation_limit/(n_parallel*nsteps*n_timepoints)))  #number of WE rounds to run

    #data collection
    aggregate_walkers = 0
    aggregate_walkers_t = []
    
    aggregate_transitions = []

    x_msm_t = []
    eqp_msm_t = []

    x_init_val = system.standard_init_coord
    x_init = np.array([x_init_val for element in range(walkers_per_bin)])
    w_init = [1/walkers_per_bin for element in range(walkers_per_bin)]

    grid = metadynamics.grid(system.standard_analysis_range, n_analysis_bins)

    for tp in range(n_timepoints):
        #run weighted ensemble with brownian dynamics
        x_init, e_init, w_init, binbounds, xtrj, etrj, wtrj, transitions, hamsm_transitions, n_trans_by_round, bincenters_flat, grid \
        = weighted_ensemble(\
                            x_init,\
                            w_init,\
                            nrounds,\
                            n_analysis_bins,\
                            walkers_per_bin,\
                            system, propagators.propagate_save1,\
                            [system, kT, dt, nsteps],\
                            system.ensemble_class,\
                            n_macrostates,\
                            grid,\
                            ha_binning=False)
        # print("------------------------------------------------------")
        # print(tp)
        # print(x_init)
        # print(w_init)
        # print(transitions)


        aggregate_walkers += len([j for i in xtrj for j in i])*nsteps
        aggregate_walkers_t.append(aggregate_walkers)
        aggregate_transitions += transitions

        x_msm, eqp_msm = MSM_methods.transitions_to_eq_probs(aggregate_transitions, bincenters_flat, show_TPM=False)
        x_msm_t.append(x_msm)
        eqp_msm_t.append(eqp_msm)

        #print(grid.grid)
        plt.plot(bincenters_flat, grid.grid)
        plt.xlabel("x-coordinate")
        plt.ylabel("metadynamics potential")

    plt.show()

    return aggregate_walkers_t, x_msm_t, eqp_msm_t, [[]]

    # #run weighted ensemble with brownian dynamics
    # #put this on multiple lines
    # x_init, e_init, w_init, binbounds, xtrj, etrj, wtrj, transitions, hamsm_transitions, n_trans_by_round, bincenters_flat \
    # = weighted_ensemble_start(\
    #                     x_init_val,\
    #                     nrounds,\
    #                     n_analysis_bins,\
    #                     walkers_per_bin,\
    #                     system, propagators.propagate_save1,\
    #                     [system, kT, dt, segment_length],\
    #                     system.ensemble_class,\
    #                     n_macrostates,\
    #                     ha_binning=False)


    #aggregate_walkers = len([j for i in xtrj for j in i])

    #build MSM
    #x_msm, eqp_msm = MSM_methods.transitions_to_eq_probs(transitions, bincenters_flat, show_TPM=False)

    #return nsteps*aggregate_walkers, x_msm, eqp_msm, []