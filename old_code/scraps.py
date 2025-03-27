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





#build an energy landscape with a specified set of minima, and transition states.
        #this landscape is to be represented by a set of points placed randomly in n-dimensional space at the specified state_density
        #temperature is used to construct the transition probability matrix (it should cancel out elsewhere? <--TODO verify this)
        #noise spectrum is the amplitude of noise to apply as a function of the spatial frequency of the noise (i.e. a low frequency noise is applied with wide gaussians)
def build_landscape(n_dim, minima_coords, minima_energies, ts_energies, state_density, kT, noise_spectrum):

    box_min = np.min(minima_coords, axis=0)
    box_max = np.max(minima_coords, axis=0)
    box_lengths = box_max-box_min
    box_padded = 3*box_lengths
    box_vol = np.prod(box_padded)
    threshold = 0.1 #TODO: set programmatically

    #obtain initial population estimates using the minima provided
    pi_all = []
    xi_all = []

    for point in range(state_density*box_vol):
        #sample uniformly distributed random coordinates within the box
        xi = np.multiply(box_padded, np.random.rand(n_dim)) + box_min - box_lengths
        
        pi = 0
        for mc, me in zip(minima_coords, minima_energies):
            pi += np.exp(-(me[0] + (np.linalg.norm(mc-xi) / me[1]**(1/n_dim))**2)/kT)

        #if pi >= threshold:
        xi_all.append(xi)
        pi_all.append(pi)

    #plt.hist(pi_all)
    #plt.show()
    print(sum(pi_all))

    plt.hist2d([i[0] for i in xi_all], [i[1] for i in xi_all], weights = pi_all, bins = [40,40], range=[[-5,5],[-5,5]])
    for mc in minima_coords:
        plt.scatter(mc[0], mc[1])

    plt.show()

    #adjust transition state energies to match system specifications by rescaling the region between each pair of gaussians
    #TODO rescale the rest of the gaussian up to keep total probability the same?
    for mci in range(len(minima_coords)):
        for mcj in range(mci+1,len(minima_coords)):
            if ts_energies[mci][mcj][1] == -1:
                continue

            print(mci)
            print(mcj)
            xts_ij = np.mean((minima_coords[mci], minima_coords[mcj]), axis = 0)

            pts_ij_i = np.exp(-(minima_energies[mci][0] + (np.linalg.norm(xts_ij-minima_coords[mci]) / minima_energies[mci][1]**(1/n_dim))**2)/kT)
            pts_ij_j = np.exp(-(minima_energies[mcj][0] + (np.linalg.norm(xts_ij-minima_coords[mcj]) / minima_energies[mcj][1]**(1/n_dim))**2)/kT)

            #ts_prob_scale_factor = np.exp(-ts_energies[mci][mcj][0]/kT)/(pts_ij_i+pts_ij_j)
            ts_enthalpy_scale_factor = ts_energies[mci][mcj][0]/(-kT*np.log(pts_ij_i+pts_ij_j))

            ts_vector = xts_ij-minima_coords[mci]

            for k, xk in enumerate(xi_all):
                xk_rel = xk-xts_ij
                xk_proj_frac = np.dot(ts_vector, xk_rel)/np.dot(ts_vector, ts_vector)
                xk_perp = xk_rel - xk_proj_frac*ts_vector
                
                pi_all[k] = pi_all[k]**(1-(1-ts_enthalpy_scale_factor)*np.exp(-(2*xk_proj_frac)**2 - ((np.linalg.norm(xk_perp)/ts_energies[mci][mcj][1]**(1/(n_dim-1)))**2)/kT))

            print(xts_ij)
            print(ts_enthalpy_scale_factor)

    plt.hist2d([i[0] for i in xi_all], [i[1] for i in xi_all], weights = pi_all, bins = [40,40], range=[[-5,5],[-5,5]])
    for mc in minima_coords:
        plt.scatter(mc[0], mc[1])

    plt.show()

    plt.scatter([i[0] for i in xi_all], [i[1] for i in xi_all], c=pi_all, cmap='viridis')


minima_coords = [[1,-1],[0,0],[2,1],[-3,0]]
#minima_coords = [[1,-1,3],[0,0,0],[2,1,1],[-3,0,1]]
minima_energies = [[1,0.9],[0,0.7],[1,0.5],[.5,1]]
ts_energies = [[[0,0],[10,.5],[0,-1],[0,-1]],[[],[0,0],[2,1],[1,50]],[[],[],[0,0],[0,-1]],[[],[],[],[0,0]]]
state_density = 100
kT = .5
noise_spectrum = "TBD"

build_landscape(2, minima_coords, minima_energies, ts_energies, state_density, kT, noise_spectrum)
