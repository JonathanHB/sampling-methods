import numpy as np
import MSM_methods
import matplotlib.pyplot as plt


#compare true and estimated energy landscapes
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


#helper function for printing mean first passage times neatly from matrices thereof
def print_mfpts_2states(mfpts, digits = 0):
    inter_well_mpfts = [mfpts[0,1], mfpts[1,0]]

    meanfmt = f"{np.mean(inter_well_mpfts):.{digits}f}"
    stdfmt = f"{np.std(inter_well_mpfts):.{digits}f}"
    print(f"MFPT = {meanfmt}+-{stdfmt} steps")



def hamsm_analysis(ha_transitions, nbins, system, save_period, lag_time=1, show_TPM=False):

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

    ha_sio_config = []
    ha_eqp_config = []
    
    for i in range(0, len(bincenters)*2, 2):
        
        ha_sio_config.append(bincenters[int(i/2)])
        ha_eqp_config.append(sum([eqp_msm[states_in_order.index(i+j)][0] if i+j in states_in_order else 0 for j in range(nm)]))


    #-----------------------------------------------------------------------------------------------------------------
    #calculate mfpts
    mfpts = MSM_methods.calc_ha_mfpts(states_in_order, eqp_msm, tpm, macrostates_discrete, nm, save_period)


    return ha_sio_config, ha_eqp_config, x_ensembles, p_ensembles, mfpts
