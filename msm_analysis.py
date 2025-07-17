import numpy as np

def bin_to_voxels_msmtrj(binbounds, msm_states, trjs):

    #-----------------------------------------------------------------------------------------
    #bin MSM states

    #bin along each dimension
    binned_trj_dim = np.array([np.digitize(trjdim, bins = binboundsdim) for trjdim, binboundsdim in zip(msm_states.transpose(), binbounds)])
        
    #flatten binned states to single index
    nbins_bydim = [len(bbd)+1 for bbd in binbounds]
    prods_higher = [np.product(nbins_bydim[i:]) for i in range(1,len(nbins_bydim))] + [1]

    bins_by_state = np.matmul(prods_higher, binned_trj_dim).transpose()

    trjs_binned = np.array([bins_by_state[trj] for trj in trjs])

    return trjs_binned

def bin_to_voxels_msmstates(binbounds, msm_states):

    #-----------------------------------------------------------------------------------------
    #bin MSM states

    #bin along each dimension
    binned_trj_dim = np.array([np.digitize(trjdim, bins = binboundsdim) for trjdim, binboundsdim in zip(msm_states.transpose(), binbounds)])
        
    #flatten binned states to single index
    nbins_bydim = [len(bbd)+1 for bbd in binbounds]
    prods_higher = [np.product(nbins_bydim[i:]) for i in range(1,len(nbins_bydim))] + [1]

    bins_by_state = np.matmul(prods_higher, binned_trj_dim).transpose()

    return bins_by_state
