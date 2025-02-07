#Jonathan Borowsky
#1/31/25
#fast row normalization based MSM construction methods; not as reliable as those in pyemma, which use fancier but slower bayesian estimators

import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components

def transitions_2_msm(transitions):
    
    #-----------------------------------------
    #ergodic trimming
    
#     if source_sink_trimming:

    #actually untrimmed; used to check if no states are removed on the first trimming
    last_trimmed_state_list = set(np.unique(transitions)) 

    for i in range(999):

        #make a list of states that have transitions both starting and ending there
        trimmed_state_list = set([tr[0] for tr in transitions]).intersection([tr[1] for tr in transitions])
        #remove all transitions where either element is not in trimmed_state_list
        transitions = [tr for tr in transitions if (tr[0] in trimmed_state_list and tr[1] in trimmed_state_list)]

        #if no transitions have been removed in the last round of trimming, 
        # we have reached a set of states which all have transitions in both directions and can proceed
        if trimmed_state_list == last_trimmed_state_list:
            break

        #update buffer used for evaluating termination condition
        last_trimmed_state_list = trimmed_state_list

        if i == 998:
            print("error: trimming failed to complete within the allotted time; please inspect data")
            return 0

    #sort states
    states_in_order = sorted(trimmed_state_list)

#     else:
#         states_in_order = sorted(np.unique(transitions))

    #-----------------------------------------
    #construct transition count matrix

    n_states = len(states_in_order)
    
    #for mapping configurational (or history-augmented) bins to MSM states
    state_to_ind = dict(zip(states_in_order, [i for i in range(n_states)]))
    
    transition_counts = np.zeros((n_states, n_states))
    
    for tr in transitions:
        if tr[0] in states_in_order and tr[1] in states_in_order:
            transition_counts[state_to_ind[tr[1]]][state_to_ind[tr[0]]] += 1
        
    #-----------------------------------------
    #connectivity trimming 
    # (part of ergodic trimming which is easier to do on the transition counts matrix)
    
    #identify the greatest connected component of the transition counts matrix, 
    # which is only normally a problem when building haMSMs
    connected_states = connected_components(transition_counts, directed=True, connection='strong')[1]
    cc_inds, cc_sizes = np.unique(connected_states, return_counts=True)
    greatest_connected_component = cc_inds[np.argmax(cc_sizes)]
        
    #remove all other components
    smaller_component_indices = [i for i, ccgroup in enumerate(connected_states) if ccgroup != greatest_connected_component]
    
    states_in_order = list(np.delete(states_in_order, smaller_component_indices))
    
    transition_counts = np.delete(transition_counts, smaller_component_indices, 0)
    transition_counts = np.delete(transition_counts, smaller_component_indices, 1)

    #I think this has a bug; it does not yield reasonable values unless sampling and WE weights are already converged
#     if symmetric:
#         transition_counts = (transition_counts + transition_counts.transpose())/2

#     plt.show()
#     plt.imshow(transition_counts)
#     plt.show()

    #-----------------------------------------
    #normalization to rate matrix
    
    #normalize transition count matrix to transition rate matrix
    #each column (aka each feature in the documentation) is normalized so that its entries add to 1, 
    # so that the probability associated with each element of X(t) is preserved 
    # (though not all at one index) when X(t) is multiplied by the TPM
    tpm = normalize(transition_counts, axis=0, norm='l1')
    
#     plt.matshow(tpm)
#     plt.show()
    
    return tpm, states_in_order


#calculate equilibrium probabilities from MSM transition probability matrix

def tpm_2_eqprobs(msm_tpm):

    #get tpm eigenvalues to find state probabilities
    msm_eigs = np.linalg.eig(msm_tpm)
        
    #make sure we're getting the correct eigenvector with eigenvalue 1
    nfigs = 12
    
    eig1_ind = -1
    
    for ex, eigenvalue in enumerate(msm_eigs[0]):
        if np.round(eigenvalue, nfigs) == 1:
            if eig1_ind != -1:
                #note that this has since been demonstrated not to occur for connected systems,
                # but serves as a warning for disconnected ones, which do have multiple 1 eigenvalues
                print("warning: multiple eigenvalues equal to 1 detected, one was selected arbitrarily")
                print(f"eigenvalues were {msm_eigs[0]}")
            eig1_ind = ex

    if eig1_ind == -1:
        print(f"error: no eigenvalue is 1 to within {nfigs} significant figures")
        
        eig1 = min(msm_eigs[0], key=lambda x:abs(x-1))
        eig1_ind = np.where(msm_eigs[0] == eig1)
        
        print(f"using eigenvalue {eig1}")

    eig0_raw = msm_eigs[1][:,eig1_ind] #this is the eigenvector associated with the eigenvalue 1
    
    #normalize so that total population = 1
    eig0 = eig0_raw/sum(eig0_raw)
    #change eigenvector to a column vector for right multiplication by tpm
    eig0c = eig0.reshape((len(eig0), 1))

    #repeatedly multiply the unrefined eigenvector by the transition probability matrix until it stops changing
    #in order to get rid of numerical errors from np.linalg.eig
    #however at times this looks like it may be chasing floating point errors
    converged = False
    refinement_rounds = 99
    maxerror = -1
            
    for r in range(refinement_rounds):
        #time evolve unrefined eigenvector
        eig0c_buffer = np.dot(msm_tpm,eig0c)

        #calculate change in state vector
        fractional_errors = (eig0c-eig0c_buffer)/eig0c
        
        maxerror = max(abs(max(fractional_errors)[0]), abs(min(fractional_errors)[0]))
        if maxerror < 10**-nfigs and min(eig0c)[0] >= 0:
            print(f"eigenvector converged to within 10^{-nfigs} after {r} rounds")
            converged = True
            break
            
        eig0c = eig0c_buffer
            
    if not converged:
        print(f"error: eigenvector failed to converge after {refinement_rounds} rounds; \
maximum fractional error of any component = {maxerror}")
    
    #some numerical inputs yield complex eigenvalues and eigenvectors but the equilibrium probability vector
    # should be real; verify that it is
    if not all(np.imag(eig0c).flatten() == np.zeros(len(eig0c))):
        print("error: nonzero complex components detected")
        
    return np.real(eig0c)