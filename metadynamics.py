import numpy as np
import itertools
import analysis
import sys


#METHOD OUTLINE
#objects:
# 1. grid: a grid with a potential in each grid cell
#      constructor: takes the output of the voxel bin construction function in analysis
#      compute_forces: returns a list of forces on atoms in each grid cell
#      add_point(x, gaussian_width, gaussian_height): adds a gaussian potential to the grid centered at x. This could also take a weight to go with gaussian_height
      
#functions:
# a propagator that takes the grid boundaries and forces from grid/compute_forces and the bin assigner from analysis and updates the positions of the atoms
# it's like a regular propagator but adds f(bin(x)) to the force

#method: run westpa with a single grid object. The object's forces are passed to the propagator at the start of each round 
# and each walker's final coordinate is added to the grid at the end of the round


class grid():

    def __init__(self, bounds, nbins_total, rate = 1, stdevs=[]):
        
        self.bounds = bounds
        self.dim = len(bounds[0])
        self.rate = rate

        if stdevs == []:
            self.stdevs = [0.025*(xmax-xmin) for xmax, xmin in zip(bounds[1], bounds[0])]
        else:
            self.stdevs = stdevs
        self.invstds = np.reciprocal(self.stdevs)

        self.bincenters, self.binwidth, self.nbins, self.nbins_total_scalar_actual, self.binbounds, self.ndim, self.prods_higher = analysis.construct_voxel_bins(bounds, nbins_total)
        
        self.grid = np.zeros(self.nbins)
        self.forcegrids = -np.gradient(self.grid, self.binwidth).reshape([self.dim]+self.nbins)

        #construct_voxel_bins should probably return this instead since it certainly doesn't need to be calculated every time this function is called
        self.grid_coords = [gci for gci in itertools.product(*[[r for r in range(nbi)] for nbi in self.nbins])]


    # Update the grid based on the new position
    def update(self, trjs, weights):

        #calculate the magnitude of the gaussian associated with every trajectory point in every grid bin
        for bc, gc in zip(self.bincenters, self.grid_coords):
            for x, w in zip(trjs, weights):

                expterm = np.multiply(np.array(bc)-np.array(x), self.invstds)
                self.grid[gc] += w * np.prod(self.invstds) * (2*np.pi)**(-self.dim/2) * np.exp(-0.5*np.dot(expterm, expterm))
    
        #updating forces here is only practical because all dimensions are part of the progress coordinate; 
        # if there were many other dimensions orthogonal to the progress coordinate, I don't think it would work
        self.forcegrids = -np.gradient(self.grid, self.binwidth).reshape([self.dim]+self.nbins)


    def compute_forces(self, trjs):

        bin_inds, nd_inds = analysis.bin_to_voxels_timeslice(self.ndim, self.binbounds, self.prods_higher, trjs)

        #see https://stackoverflow.com/questions/69865664/get-values-from-an-array-given-a-list-of-coordinates-locations-with-numpy
        forces = np.array([fgc.take(np.ravel_multi_index(nd_inds.T, fgc.shape)) for fgc in self.forcegrids]).transpose()

        return forces