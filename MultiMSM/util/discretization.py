'''
The file contains utility objects for discretizing the free variable for the MSM
construction, in my case the monomer fraction. 

Contains a Discretization object that stores the discretization bins and can determine
what interval a point lies in. 

Also includes an optimizer to choose the "best" discretization according to a user 
provided metric. Starting choices are to minimze the distance between a given structure
yield curve, and the sampling approximation to that curve. Performs a Monte Carlo search
over bin placements at fixed number of bins. 
'''
import numpy as np

from MultiMSM.MSM.MultiMSM import Collection
from MultiMSM.util.sampling_data import ClusterSizeData


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
    

class BinOptimizer:

    def __init__(self, num_bins, lag, MM, traj_folder, initial_guess = None,
                 fixed_indices = None,
                 max_iter = 500, beta = 1.0, step_size = 0.05):

        #set user inputs
        self.__num_bins    = num_bins
        self.__lag         = lag
        self.__MM          = MM
        self.__traj_folder = traj_folder

        #check for initial guess and right type
        if initial_guess is not None:
            if isinstance(initial_guess, Discretization):
                self.__initial_guess = initial_guess
                self.__fixed_indices = []
                if fixed_indices is not None:
                    self.__fixed_indices = fixed_indices

            else:

                err_msg = "Please provide a Discretization object for the initial guess"
                err_msg+= " or omit it to use default equally spaced grids"
                raise TypeError(err_msg)
            
        #use default equally spaced grid 
        else:
            disc = Discretization()
            disc.set_equally_spaced(self.__num_bins+1)
            self.__initial_guess = disc 

        #variables for the target state
        self.__target_size = 1
        self.__apply_target_settings()

        #variables for the optimization
        self.__max_iter = max_iter
        self.__beta     = beta
        self.__step_size= step_size

        #variables for the output
        self.__best_obj  = 1000
        self.__best_disc = None 


        return
    
    def set_target(self, size):
        #set the intermediate size to minimze distance for. apply the settings

        self.__target_size = size
        self.__apply_target_settings()
        return
    
    def __apply_target_settings(self):
        #compute the exact solution for the given target, and the corresponding indices
        #in the Markov state model

        #get the "exact" solution from sampling data
        S = ClusterSizeData(self.__traj_folder, recompute=False)
        dists = S.get_normalized_distribution(mass_weighted=True)
        self.__exact = dists[:,self.__target_size-1]

        #get the length of sampling data to know how many lags to go for
        self.__num_lags = len(self.__exact)

        #get the indices of the correspondind markov states
        self.__target_indices = self.__MM.filter_by_size(self.__target_size)

        return
    
    def run_optimization(self):

        #TODO: 
        #get the current energy


        return

    def __eval_energy(self, model):

        final_time   = round(self.__num_lags/self.__lag)
        soln         = model.get_FKE(T=final_time)
        target_prob  = soln[:,self.__target_indices].sum(1)

        rel_error = np.zeros(self.__num_lags, dtype=float)
        for i in range(self.__num_lags):

            abs_error = target_prob[i]-self.__exact[i]
            if self.__exact[i] > 1e-5:
                rel_error[i] = (abs_error/self.__exact[i])
            else:
                rel_error[i] = (abs_error)

        return np.linalg.norm(rel_error,2)
    
    def __propose_move(self, current_guess):
        #proposal move for MC search. randomly shift a random cutoff

        #make a new Disc object and copy the current
        proposal = Discretization(current_guess.get_cutoffs())

        #choose a bin index to move - b/w 1 and num_bins-1 inclusive
        index = self.__choose_random_integer(1, self.__num_bins-1)

        #choose an amount to move it - range [-step_size,step_size]
        random_shift = (2 * np.random.random() - 1) * self.__step_size

        #apply the perturbation
        proposal.perturb(index, random_shift)

        return proposal

    def __choose_random_integer(self, lower, upper):
        #choose a random integer between lower and upper
        #check that it is not forbidden by the fixed list

        max_tries = 500

        for i in range(max_tries):

            r_int = np.random.randint(lower, upper+1)
            if r_int not in self.__fixed_indices:
                return r_int
            
        #print error message if we cannot choose an index
        err_msg = "Could not choose a valid random cutoff in 500 tries. "
        err_msg+= "Are there enough degrees of freedom compared to fixed values?"
        raise RuntimeError(err_msg)
        return
    
    def check_accept(self, energy):






