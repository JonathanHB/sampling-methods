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