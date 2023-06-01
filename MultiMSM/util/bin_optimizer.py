'''
Implements an optimizer to choose the "best" discretization for a MultiMSM according to 
a user provided metric. Starting choices are to minimze the distance between a given 
structure yield curve, and the sampling approximation to that curve. Performs a Monte 
Carlo search over bin placements at fixed number of bins. 
'''

import numpy as np

from MultiMSM.MSM.MultiMSM import MultiMSMBuilder
from MultiMSM.util.sampling_data import ClusterSizeData
from MultiMSM.util.discretization import Discretization



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
        #perform the MC optimization

        #start by constructing the initial model and getting an energy
        model0 = MultiMSMBuilder(self.__initial_guess, self.__MM, self.__traj_folder,
                                  self.__lag).make()
        E0 = self.__eval_energy(model0)
        print(E0)



        return

    def __eval_energy(self, model):

        final_time   = round(self.__num_lags/self.__lag)
        soln         = model.get_FKE(T=final_time, p0="monomer_start")
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

        return
    

