import numpy as np
import matplotlib.pyplot as plt

#superclass of 1d potential functions
class potential_well_1d():

    def __init__(self, potential, macro_class):
        self.potentiall = potential
        self.macro_classs = macro_class

    def ensemble_class(self, x, e):  
        ms = self.macro_classs(x)
        if ms != -1:
            return ms
        else:
            return e

    def normalized_pops_energies(self, kT, bincenters):
        #assume equal bin widths
        binwidth = bincenters[1]-bincenters[0]

        pops_nonnorm = [np.exp(-self.potentiall(x)/kT) for x in bincenters]
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
            energy_gap = abs(self.potentiall(bin_boundaries[i+1])-self.potentiall(bin_boundaries[i]))/kT
            n_subbins = max(int(np.ceil(energy_gap/tolerance)), 1)
            
            bin_pop = 0
            subbin_width = (bin_boundaries[i+1] - bin_boundaries[i])/n_subbins
            
            for sbx in np.linspace(bin_boundaries[i]+subbin_width/2, bin_boundaries[i+1]-subbin_width/2, n_subbins):
                bin_pop += np.exp(-self.potentiall(sbx)/kT)*subbin_width
                
            bin_populations.append(bin_pop)

        z = sum(bin_populations)
        bin_populations = [bp/z for bp in bin_populations]
        
        return bin_centers, bin_populations


class unit_double_well(potential_well_1d):

    def potential(self, x):
        return x**4 - x**2
        
    def F(self, x):
        return -4*x**3 + 2*x

    #for visualization to check that you've written the potential right
    def plot_quantity(self, quantity):
        x_extr = 1.5
        plt.plot(np.linspace(-x_extr, x_extr, 100), [quantity(i) for i in np.linspace(-x_extr, x_extr, 100)])

    def diffusion_coefficient(self):
        return 1
    
    def macro_class(self, x):
        thr = 1/np.sqrt(2)
        if x <= -thr:
            return 0
        elif x >= thr:
            return 1
        else:
            return -1
            
    def n_macrostates(self):
        return 2
    
    def standard_init_coord(self):
        return -1/np.sqrt(2) 

    def __init__(self):
        super().__init__(self.potential, self.macro_class)


class unit_sine_well(potential_well_1d):

    def potential(self, x):
        return 0.0001*x**4 + np.cos(x)
        
    def F(self, x):
        return 0.0001*-4*x**3 - np.sin(x)

    #for visualization to check that you've written the potential right
    def plot_quantity(self, quantity): 
        x_extr = 20
        plt.plot(np.linspace(-x_extr, x_extr, 100), [quantity(i) for i in np.linspace(-x_extr, x_extr, 100)])

    def diffusion_coefficient(self):
        return 1
    
    def macro_class(self, x):
        thr = 2*np.pi
        if x < -thr:
            return 0
        elif x > thr:
            return 1
        else:
            return -1
            
    def n_macrostates(self):
        return 2
    
    def standard_init_coord(self):
        return -3*np.pi

    def __init__(self):
        super().__init__(self.potential, self.macro_class)

