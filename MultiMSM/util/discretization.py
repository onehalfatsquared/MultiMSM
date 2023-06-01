'''
The file contains utility objects for discretizing the free variable for the MSM
construction, in my case the monomer fraction. 

Contains a Discretization object that stores the discretization bins and can determine
what interval a point lies in. 
'''
import numpy as np

class Discretization:

    '''
    Discretize the monomer fraction (value in [0,1]) according to user criteria.
    Methods will take in a value and determine which discretized set that value is part of. 
    '''

    def __init__(self, cutoffs = None):

        #if no cutoff is provided, construct a single MSM
        if cutoffs is None:

            default_cutoffs = [0,1]
            self.set_cutoffs(default_cutoffs)
        
        else:

            self.__check_cutoffs(cutoffs)
            self.set_cutoffs(cutoffs)

        return
    
    def get_cutoffs(self):

        return self.__interval_cutoffs

    def set_cutoffs(self, cutoffs):

        self.__interval_cutoffs = cutoffs
        return

    def set_equally_spaced(self, num_points):
        #discretize the interval [0,1] with given number of points

        self.__interval_cutoffs = np.linspace(0, 1, num_points)
        print(self.__interval_cutoffs)
        return

    def get_num_intervals(self):

        return len(self.__interval_cutoffs) - 1

    def determine_interval(self, value):

        #note: this will be called many times (1 per sample). can be potentially sped up (numba?)
        return np.digitize(value+1e-8, self.__interval_cutoffs)

    def interval_index_to_bounds(self, interval_num):

        lower = round(self.__interval_cutoffs[interval_num-1],3)
        upper = round(self.__interval_cutoffs[interval_num],3)
        return (lower, upper)
    
    def perturb(self, index, amount):
        #modify the location of index bin by amount
        #must remain within the bounds of the discretization

        new_value = self.__interval_cutoffs[index] + amount
        if new_value < 0.05:
            new_value = 0.05
        elif new_value > 0.95:
            new_value = 0.95

        self.__interval_cutoffs[index] = new_value

        #sort in case the cutoff went passed the next bin
        self.__interval_cutoffs.sort()
        return

    def __check_cutoffs(self, cutoffs):
        #check that the supplied cutoffs form a valid discretization of [0,1]

        tol = 1e-4

        if abs(cutoffs[0]) > tol:
            err_msg  = "The supplied discretization does not start at 0.\n"
            raise RuntimeError(err_msg)

        if abs(cutoffs[0]) > tol:
            err_msg  = "The supplied discretization does not start at 0.\n"
            raise RuntimeError(err_msg)

        if cutoffs != sorted(cutoffs):
            err_msg  = "The discretization is not sorted.\n"
            raise RuntimeError(err_msg)

    def __len__(self):

        return self.get_num_intervals()
    

