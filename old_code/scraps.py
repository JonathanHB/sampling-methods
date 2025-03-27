import numpy as np

#unused; deprecated
def propagate(trj_coords, system, kT, timestep, nsteps):

    D = system.diffusion_coefficient()
    
    B = lambda x : x + D/kT * system.F(x) * timestep + np.sqrt(2*D*timestep)*np.random.normal()
    
    for step in range(nsteps):
        trj_coords = list(map(B, trj_coords))

    return trj_coords


# def get_bin_boundaries(trj_flat, nbins, binrange = [], symmetric = True):

#     if binrange == []:

#         epsilon = 10**-9 #to avoid any >= vs > issues at bin boundaries
#         if symmetric:    
#             bin_extreme = max(np.max(trj_flat), -np.min(trj_flat))
#             bin_min = -bin_extreme-epsilon
#             bin_max = bin_extreme+epsilon
#         else:
#             bin_min = np.min(trj_flat)-epsilon
#             bin_max = np.max(trj_flat)+epsilon
#     else:
#         bin_min = binrange[0]
#         bin_max = binrange[1]

#     step = (bin_max-bin_min)/nbins
    
#     binbounds = np.linspace(bin_min, bin_max, nbins+1)
#     bincenters = np.linspace(bin_min-step/2, bin_max+step/2, nbins+2)

#     return binbounds, bincenters, step


#DEPRECATED; superseded by methods in analysis.py
def landscape_recovery(xtrj, wtrj, binbounds, transitions, hamsm_transitions, n_trans_by_round, t, n_macrostates, potential_func, macrostate_classifier, kT):
    
    binwidth = (binbounds[-1]-binbounds[0])/len(binbounds)
    bincenters = np.linspace(binbounds[0]-binwidth/2, binbounds[-1]+binwidth/2, len(binbounds)+1)
    
    #--------------------------------------------
    #energy estimate from WE weights
    xtrj_flat = [j for i in xtrj[0:t] for j in i ]
    wtrj_flat = [j for i in wtrj[0:t] for j in i ]

    binned_trj = np.digitize(xtrj_flat, bins = binbounds)

    #+1 is for the end bins
    binned_total_weights = np.zeros(len(binbounds)+1)
    for i, b in enumerate(binned_trj):
        binned_total_weights[b] += wtrj_flat[i]/t
        
    sampled_we_inds_energies = [[i, -np.log(wt/binwidth)] for i, wt in enumerate(binned_total_weights) if wt > 0]
    sampled_we_bincenters = [bincenters[wie[0]] for wie in sampled_we_inds_energies]
    sampled_we_energies = [wie[1] for wie in sampled_we_inds_energies]
    
    return bincenters, binned_total_weights, sampled_we_bincenters, sampled_we_energies



#DEPRECATED
#non-history-augmented and is probably also slightly wrong; the haMSM formulation should be used instead
def calc_MFPT(tpm, x_msm, eqp_msm_init, macrostate_classifier, n_macrostates, lag_time, save_period):
    
    msm_state_macrostates = [macrostate_classifier(x) for x in x_msm]

    mfpts = np.zeros([n_macrostates, n_macrostates])

    #calculate MFPT into each destination macrostate
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

                    #calculate how much probability is left in the initial macrostate
                    prob_in_init = sum([eqpi[0] if msm_state_macrostates[i] == mi else 0 for i, eqpi in enumerate(eqp_msm)]) #there's a faster way of doing this
                    rate_per_unit.append((p_t[-2]-p_t[-1])/prob_in_init)

                mfpts[mf][mi] = save_period*lag_time/rate_per_unit[-1]
                #plt.plot(rate_per_unit)
                #plt.show()

    return mfpts


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