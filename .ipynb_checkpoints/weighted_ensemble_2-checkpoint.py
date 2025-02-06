#Method for running weighted ensemble

import numpy as np
import random
import sys
import matplotlib.pyplot as plt

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

def weighted_ensemble(x_init, w_init, nrounds, nbins, walkers_per_bin, binrange, propagator, prop_params, macrostate_classifier, n_macrostates, ha_binning=False):

    split_limit = 2.00001*sys.float_info.min #0.0002
    merge_limit = 1 #effectively no limit
    
    #n_macrostates = 2
    
    #extreme_val = 10**6
    #binbounds = np.concatenate(([-extreme_val], np.linspace(binrange[0], binrange[1], nbins+1), [extreme_val]))
    #if not ha_binning:
    binbounds = np.linspace(binrange[0], binrange[1], nbins+1)
    #else:
    #    binbounds = np.linspace(binrange[0], binrange[1], (nbins+1)*n_macrostates)
    
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
        config_bin_inds = np.digitize(x_init, binbounds)
        
        if not ha_binning:
            bin_inds = config_bin_inds
            inds_by_bin = [[] for element in range(nbins+2)]

        else:
#             print(max(config_bin_inds))
            bin_inds = [cbi*n_macrostates+e_init[einit_ind] for einit_ind, cbi in enumerate(config_bin_inds)]
            inds_by_bin = [[] for element in range((nbins+2)*n_macrostates)]

        #the indices in x_init of the walkers in each bin
#         print(len(inds_by_bin))
#         print(max(bin_inds))
        for xinit_ind, bin_ind in enumerate(bin_inds):
            #print(bin_ind)
            inds_by_bin[bin_ind].append(xinit_ind)
            
            
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
                w_indset = [w_init[i] if w_init[i] != 0 else sys.float_info.min for i in indset]

                #duplicated_walkers = random.choices(indset, weights=w_indset, k = walkers_per_bin-len(indset))

                #always split the heaviest walker
                walker_to_split = np.argmax(w_indset)
                
                #add coordinates and weights of walkers from this bin to the list for next round
                # coordinates are unchanged for duplicated walkers; weights are reduced
                for i in indset:
                    x_md.append(x_init[i])
                    e_md.append(e_init[i])                    

                    if i == walker_to_split and max(w_indset) >= split_limit:
                        #add halved weight for first child
                        w_md.append(w_init[i]/2)

                        #add second child walker
                        x_md.append(x_init[i])
                        e_md.append(e_init[i])
                        w_md.append(w_init[i]/2)

                    else:
                        w_md.append(w_init[i])
            
                    # #add multiple copies of walkers to be duplicated with proportionally smaller weights
                    # for j in range(1+duplicated_walkers.count(i)):
                    #     x_md.append(x_init[i])
                    #     e_md.append(e_init[i])

                    #     if w_init[i] >= 0.002:
                    #         #this is the normal WE algorithm
                    #         w_md.append(max(w_init[i]/(1+duplicated_walkers.count(i)), sys.float_info.min))
                    #     else:
                    #         w_md.append(w_init[i])
                    #         break #do not duplicate too-light walkers
                            
            #merge simulations in bins with too many walkers
            elif len(indset) > walkers_per_bin:
                #instead of doing what's below just merge the two lightest walkers to prevent probability from accumulating in heavier ones
                
                w_indset = [w_init[i] for i in indset]
                weights_ranked = list(np.argsort(w_indset))
                
                #note that using argmin and index will yield different results when multiple walkers have the same weight
                ind_lightest = weights_ranked.index(0)  
                weight_lightest = w_indset[ind_lightest]
                
                ind_second_lightest = weights_ranked.index(1)
                weight_second_lightest = w_indset[ind_second_lightest]

                #remove no walker if none meet the criteria below
                removed_walker = -1
                
                if weight_lightest < merge_limit:
                
                    weights_pair = weight_lightest + weight_second_lightest
                    inds_removal = [ind_lightest, ind_second_lightest]
                    weights_removal = [weight_lightest/weights_pair, weight_second_lightest/weights_pair]
    
                    removed_walker = random.choices(inds_removal, weights=weights_removal, k = 1)[0]

                for ii, i in enumerate(indset):
                    if ii != removed_walker:
                        x_md.append(x_init[i])
                        e_md.append(e_init[i])

                        #add the removed walker's weight to the walker with which it was merged
                        if ii in [ind_lightest, ind_second_lightest] and removed_walker != -1:
                            w_md.append(w_init[i] + w_init[indset[removed_walker]])
                        else:
                            w_md.append(w_init[i])

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
        
        #bin transitions and add them to the transition list
        bin_inds_1 = np.digitize(x_md, binbounds) #not the same as bin_inds
        bin_inds_2 = np.digitize(x_init, binbounds)
        
        transitions +=  [[b1,b2] for b1,b2 in zip(bin_inds_1, bin_inds_2)]
        hamsm_transitions += [[b1*n_macrostates+e1,b2*n_macrostates+e2] for b1,b2,e1,e2 in zip(bin_inds_1, bin_inds_2, e_md, e_init)]
        
        n_trans_by_round.append(len(transitions))

        
    return x_init, e_init, w_init, binbounds, xtrj, etrj, wtrj, transitions, hamsm_transitions, n_trans_by_round


def weighted_ensemble_start(x_init_val, nrounds, nbins, walkers_per_bin, binrange, propagator, prop_params, macrostate_classifier, n_macrostates, ha_binning=False):

    #start 1 bin worth of walkers at x_init_val with equal weights
    x_init = np.array([x_init_val for element in range(walkers_per_bin)])
    w_init = [1/walkers_per_bin for element in range(walkers_per_bin)]
    
    #run weighted ensemble with brownian dynamics
    #put this on multiple lines
    return weighted_ensemble(\
                        x_init,\
                        w_init,\
                        nrounds,\
                        nbins,\
                        walkers_per_bin,\
                        binrange, propagator,\
                        prop_params,\
                        macrostate_classifier,\
                        n_macrostates,\
                        ha_binning=False)


def landscape_recovery(xtrj, wtrj, binbounds, transitions, hamsm_transitions, n_trans_by_round, t, n_macrostates, potential_func, macrostate_classifier, kT):
    
    binwidth = (binbounds[-1]-binbounds[0])/len(binbounds)

    # bin energies integrated over the width of the bin,
    # so the energies below should be plotted at the centers of the corresponding bins
    bincenters = [binbounds[0]-binwidth/2] + [bb+binwidth/2 for bb in binbounds]

    #--------------------------------------------
    #true energy
    implied_pops_nonnorm = [np.exp(-potential_func(x)/kT) for x in bincenters]
    total_nonnorm_pop = sum(implied_pops_nonnorm)
    pops_norm = [p/total_nonnorm_pop for p in implied_pops_nonnorm]
    
    energies_norm = [-kT*np.log(p/(total_nonnorm_pop*binwidth)) for p in implied_pops_nonnorm]
    #plt.plot(bincenters, energies_norm)
    
    #--------------------------------------------
    #energy estimate from WE weights
    xtrj_flat = [j for i in xtrj[0:t] for j in i ]
    wtrj_flat = [j for i in wtrj[0:t] for j in i ]

    binned_trj = np.digitize(xtrj_flat, bins = binbounds)

    #+2 is for the end bins
    binned_total_weights = np.zeros(len(binbounds)+1)
    for i, b in enumerate(binned_trj):
        binned_total_weights[b] += wtrj_flat[i]/t
        
    we_i_energies = [[i, -np.log(wt/binwidth)] for i, wt in enumerate(binned_total_weights) if wt > 0]
    we_bincenters = [bincenters[wie[0]] for wie in we_i_energies]
    we_energies = [wie[1] for wie in we_i_energies]
    
    #plt.plot(we_bincenters, we_energies, linestyle="dashed")
    #plt.show()

    #probability density comparison
    plt.plot(bincenters, pops_norm)
    plt.plot(bincenters, binned_total_weights)
    plt.show()
    
    rmse_weighted = np.sqrt(np.mean([epa*(eps-epa)**2 for epa, eps in zip(pops_norm, binned_total_weights)]))
    #kl_divergence = sum([epa*np.log(epa/eps) for epa, eps in zip(pops_norm, binned_total_weights)])
    
    #print(f"kl divergence = {kl_divergence}")
    print(f"weighted RMSE = {rmse_weighted}")
