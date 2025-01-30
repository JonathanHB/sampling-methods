#Method for running weighted ensemble

import numpy as np

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

                duplicated_walkers = random.choices(indset, weights=w_indset, k = walkers_per_bin-len(indset))
                
                #add coordinates and weights of walkers from this bin to the list for next round
                # coordinates are unchanged for duplicated walkers; weights are reduced
                for i in indset:
                    #add multiple copies of walkers to be duplicated with proportionally smaller weights
                    for j in range(1+duplicated_walkers.count(i)):
                        x_md.append(x_init[i])
                        e_md.append(e_init[i])
                        w_md.append(max(w_init[i]/(1+duplicated_walkers.count(i)), sys.float_info.min))

            #merge simulations in bins with too many walkers
            elif len(indset) > walkers_per_bin:

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
        x_init = propagator(x_md, prop_params[0], prop_params[1], prop_params[2], prop_params[3], prop_params[4])
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
        