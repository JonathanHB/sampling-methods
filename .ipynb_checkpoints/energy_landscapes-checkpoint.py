import numpy as np
import matplotlib.pyplot as plt

class unit_double_well():
    
    def potential(self, x):
        return x**4 - x**2
        
    def F(self, x):
        return -4*x**3 + 2*x

    def plot_quantity(self, quantity):
        x_extr = 1.5
        plt.plot(np.linspace(-x_extr, x_extr, 100), [quantity(i) for i in np.linspace(-x_extr, x_extr, 100)])

    def diffusion_coefficient(self):
        return 1
    
    def macro_class(self, x):
        thr = 1/np.sqrt(2)
        if x < -thr:
            return 0
        elif x > thr:
            return 1
        else:
            return -1
            
    def ensemble_class(self, x, e):  
        ms = self.macro_class(x)
        if ms != -1:
            return ms
        else:
            return e

    def n_macrostates(self):
        return 2
    
    def standard_init_coord(self):
        return -1/np.sqrt(2)

    #tolerance is the permitted energy difference between the edges of a sub-bin in kT
    #bin boundaries are assumed to increase monotonically
    #the increase in accuracy provided by this method seems to be entirely unnecessary
    def compute_true_populations(self, bin_boundaries, kT, tolerance = 0.01):

        bin_centers = []
        bin_populations = []

        for i in range(len(bin_boundaries)-1):
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1])/2)

            #figure out how many sub-bins the bin must be divided in for the potential across each bin to be roughtly constant
            #assuming negligible curvature; this method will fail for bins with equal edge heights which curve up or down in between
            #a more general approach would be to randomly sample points in each bin and then average or sum somehow
            energy_gap = abs(self.potential(bin_boundaries[i+1])-self.potential(bin_boundaries[i]))/kT
            n_subbins = max(int(np.ceil(energy_gap/tolerance)), 1)
            
            bin_pop = 0
            subbin_width = (bin_boundaries[i+1] - bin_boundaries[i])/n_subbins
            
            for sbx in np.linspace(bin_boundaries[i]+subbin_width/2, bin_boundaries[i+1]-subbin_width/2, n_subbins):
                bin_pop += np.exp(-self.potential(sbx)/kT)*subbin_width
                
            bin_populations.append(bin_pop)

        z = sum(bin_populations)
        bin_populations = [bp/z for bp in bin_populations]
        
        return bin_centers, bin_populations
        

def landscape_comparison(system, kT, coordinates, estimated_populations, metrics = []):

    #compute true populations 
    z_kT = sum([np.exp(-system.potential(x)/kT) for x in coordinates])
    eq_pops_analytic = [np.exp(-system.potential(x)/kT)/z_kT for x in coordinates]

    #unnecessary extra precision; requires bin boundaries as an argument
    #bincenters, eq_pops_analytic = system.compute_true_populations(bins, kT, tolerance = 0.01)
    #plt.plot(bincenters, eq_pops_analytic)
    
    plt.plot(coordinates, eq_pops_analytic, linestyle="dashed")
    plt.plot(coordinates, eq_pops_simulation)
    
    plt.show()

    if "rmsew" in metrics:
        #without weighting, extending the region over which the RMS error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        rmse_weighted = np.sqrt(np.mean([epa*(eps-epa)**2 for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)]))
        print(f"weighted RMSE = {rmse_weighted}")

    if "maew" in metrics:
        #without weighting, extending the region over which the mean absolute error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        mae_weighted = np.mean([epa*abs(eps-epa) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])
        print(f"weighted MAE = {mae_weighted}")            


    #kl divergence has bad numerical properties when any estimated entries are 0, which some usually are
    #kl_divergence = sum([epa*np.log(epa/eps) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])
    #print(f"kl divergence = {kl_divergence}")
    