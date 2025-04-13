import numpy as np
import deeptime



#build an energy landscape with a specified set of minima, and transition states.
        #this landscape is to be represented by a set of points placed randomly in n-dimensional space at the specified state_density
        #temperature is used to construct the transition probability matrix (it should cancel out elsewhere? <--TODO verify this)
        #noise spectrum is the amplitude of noise to apply as a function of the spatial frequency of the noise (i.e. a low frequency noise is applied with wide gaussians)
def build_landscape(n_dim, minima_coords, minima_energies, state_density, kT, threshold, min_spacing, noise_spectrum):

    #select box dimensions that will include the whole of the relevant landscape
    mc_max = []
    mc_min = []

    #find dimensions that will extend 3 standard deviations from the bottom of each well
    for mci, mei in zip(minima_coords, minima_energies):
        mc_min.append([mcij-3*mei[1] for mcij in mci])
        mc_max.append([mcij+3*mei[1] for mcij in mci])

    box_min = np.min(mc_min, axis=0)
    box_max = np.max(mc_max, axis=0)

    box_lengths = box_max-box_min
    box_vol = np.prod(box_lengths)

    #obtain initial population estimates using the minima provided
    pi_all = []
    xi_all = []

    for point in range(int(round(state_density*box_vol))):
        #sample uniformly distributed random coordinates within the box
        xi = np.multiply(box_lengths, np.random.rand(n_dim)) + box_min
        
        #keep only coordinates that are sufficiently far from existing ones since a more uniform distribution is a more efficient representation of configuration space
        #note that this creates surfaces slightly denser than the interior of the populated regions because there are no neighbors on one side which could be below the spacing threshold
        #but avoiding this would be expensive and it probably doesn't matter, to paraphrase the last words of many unfortunate souls.
        if len(xi_all) > 0:
            #print(np.stack(xi_all))
            #print(np.stack([xi for l in range(len(xi_all))], axis=0))
            dists_to_existing = np.linalg.norm(np.stack([xi for l in range(len(xi_all))], axis=0) - np.stack(xi_all), axis=1)
            if np.min(dists_to_existing) < min_spacing:
                point -= 1
                continue

        #count up total probability at current coordinates from all of the harmonic wells provided as arguments
        pi = 0
        for mc, me in zip(minima_coords, minima_energies):
            pi += np.exp(-(me[0] + (np.linalg.norm(mc-xi) / me[1]**(1/n_dim))**2))

        #discard nearly empty wells to save time
        if pi >= threshold:
            xi_all.append(xi)
            pi_all.append(pi)

    #normalize probabilities
    p_tot = sum(pi_all)
    pi_all_ref_t = [pii/p_tot for pii in pi_all]
    
    #calculate energies and then determine populations at a temperature differing by a factor of kT from the reference temperature of 1
    e_all = [-np.log(pii) for pii in pi_all_ref_t]
    pi_all_kt = [np.exp(-eii/kT) for eii in e_all]
    p_tot_kt = sum(pi_all_kt)
    pi_all_kt = [pii/p_tot_kt for pii in pi_all_kt]
    
    return np.stack(xi_all), pi_all_ref_t, e_all, pi_all_kt, box_min, box_max



#build a MSM representing a synthetic energy landscape with energies e_all at points xi_all
def synthetic_msm(xi_all, e_all, min_spacing, kT):

    trm = np.zeros([len(e_all), len(e_all)])

    r1 = 0.1

    for i, ei in enumerate(e_all):
        for j, ej in enumerate(e_all):
            #add transitions between nearby points. Transitions between distant points would have negligible rates and have been omitted
            if np.linalg.norm(xi_all[i]-xi_all[j]) <= 2*min_spacing:
                if ei > ej:
                    trm[i,j] = (r1**(np.linalg.norm(xi_all[i]-xi_all[j])/min_spacing)) * np.exp(-(ei-ej)/kT)
                else:
                    trm[i,j] = r1**(np.linalg.norm(xi_all[i]-xi_all[j])/min_spacing)
                    
    #set self transition probabilities so that each column is normalized
    for k in range(len(e_all)):
        trm[k,k] = 0
        trm[k,k] = 1-sum(trm[:,k])
        
    #plt.imshow(trm)
    #plt.show()

    dtmsm = deeptime.markov.msm.MarkovStateModel(trm.transpose(), stationary_distribution=None, reversible=None, n_eigenvalues=None, ncv=None, count_model=None, transition_matrix_tolerance=1e-08, lagtime=None)

    return dtmsm


class two_wells_decoy_valley():

    def __init__(self):
        
        self.minima_coords   = [[0,0],[8,0],[0,1],  [1,1],  [1,2],  [2,2],  [3,2],  [4,2],  [5,2],  [6,2],  [7,2],  [7,1],  [8,1],  [0,-1], [1,-1], [1,-2], [2,-2], [3,-2], [4,-2], [5,-2]]
        self.minima_energies = [[0,1],[0,1],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[2,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3],[1,0.3]]
        self.state_density = 200
        self.kT = .4
        self.noise_spectrum = "TBD"
        self.threshold = 0.01 #note that this is a population threshold at the reference temperature, not an energy threshold
        self.min_spacing = 0.3
        self.n_dim = 2

        #find the average distance to the 7th or 8th nearest neighbor and use that as a threshold for MSM construction
        #set the base timescale to be very short so that only the nearest neighbors matter and then multiply that MSM by itself to get a TPM for timescales of interest
        #employ sparse matrices for efficient propagation

        self.x, self.p_ref_temp, self.e, self.p, self.box_min, self.box_max = build_landscape(self.n_dim, self.minima_coords, self.minima_energies, self.state_density, self.kT, self.threshold, self.min_spacing, self.noise_spectrum)
        self.dtmsm = synthetic_msm(self.x, self.e, self.min_spacing, self.kT)

        self.start_from_index = True
        self.standard_init_ind = np.argmin(self.e)
        self.standard_init_coord = self.x[self.standard_init_ind]
        self.standard_analysis_range = [self.box_min, self.box_max]


