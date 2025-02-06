import numpy as np
import matplotlib.pyplot as plt

#make a 1d_system superclass and then make systems with different potential functions and macrostate classifiers subclasses that inherit its methods
class unit_double_well():
    
    def potential(self, x):
        return x**4 - x**2
        
    def F(self, x):
        return -4*x**3 + 2*x

    def plot_quantity(self, quantity): #only useful for visualization to check that you've written the potential right; do not put in superclass
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


    def normalized_pops_energies(self, kT, bincenters):
        #assume equal bin widths
        binwidth = bincenters[1]-bincenters[0]

        pops_nonnorm = [np.exp(-self.potential(x)/kT) for x in bincenters]
        z = sum(pops_nonnorm)
        pops_norm = [p/z for p in pops_nonnorm]
    
        energies_norm = [-kT*np.log(p/(z*binwidth)) for p in pops_nonnorm]

        return pops_norm, energies_norm

    #compute equilibrium populations of the given bins by integrating across them
    # this should be more accurate than just using the center point of the bin
    # the increase in accuracy provided by this method seems to be entirely unnecessary in practice
    #tolerance is the permitted energy difference between the edges of a sub-bin in kT
    #bin_boundaries are assumed to increase monotonically
    def compute_true_populations(self, bin_boundaries, kT, tolerance = 0.01):

        bin_centers = []
        bin_populations = []

        for i in range(len(bin_boundaries)-1):
            bin_centers.append((bin_boundaries[i] + bin_boundaries[i+1])/2)

            #figure out how many sub-bins the bin must be divided into for the potential across each bin to be roughtly constant
            #This method assumes negligible curvature and will fail for bins with equal edge energies which curve up or down in between
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
        

#compare true and estimated energy landscapes
def landscape_comparison(system, kT, coordinates, eq_pops_simulation, metrics = []):

    #compute true populations 
    eq_pops_analytic, energies_analytic = system.normalized_pops_energies(kT, coordinates)

    #unnecessary extra precision; requires bin boundaries as an argument
    #bincenters, eq_pops_analytic = system.compute_true_populations(bins, kT, tolerance = 0.01)
    #plt.plot(bincenters, eq_pops_analytic)
    
    plt.plot(coordinates, eq_pops_analytic, linestyle="dashed")
    plt.plot(coordinates, eq_pops_simulation)
    
    plt.show()

    metrics = {}

    if "rmsew" in metrics:
        #without weighting, extending the region over which the RMS error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        rmse_weighted = np.sqrt(np.mean([epa*(eps-epa)**2 for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)]))
        print(f"weighted RMSE = {rmse_weighted}")
        metrics["rmsew"] = rmse_weighted

    if "maew" in metrics:
        #without weighting, extending the region over which the mean absolute error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        mae_weighted = np.mean([epa*abs(eps-epa) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])
        print(f"weighted MAE = {mae_weighted}")            
        metrics["maew"] = rmse_weighted

    #kl divergence has bad numerical properties when any estimated entries are 0, which some usually are
    #kl_divergence = sum([epa*np.log(epa/eps) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])
    #print(f"kl divergence = {kl_divergence}")


    