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

import matplotlib.pyplot as plt



class BinOptimizer:

    def __init__(self, num_bins, lag, MM, traj_folder, initial_guess = None,
                 fixed_indices = None):

        #set user inputs
        self._num_bins    = num_bins
        self._lag         = lag
        self._MM          = MM
        self._traj_folder = traj_folder

        #check for initial guess and right type
        if initial_guess is not None:
            if isinstance(initial_guess, Discretization):
                self._initial_guess = initial_guess
                self._fixed_indices = []
                if fixed_indices is not None:
                    self._fixed_indices = fixed_indices

            else:

                err_msg = "Please provide a Discretization object for the initial guess"
                err_msg+= " or omit it to use default equally spaced grids"
                raise TypeError(err_msg)
            
        #use default equally spaced grid 
        else:
            disc = Discretization()
            disc.set_equally_spaced(self._num_bins+1)
            self._initial_guess = disc 
            self._fixed_indices = []

        #variables for the target state - init as monomer
        self._target_size = 1
        self._apply_target_settings()

        #variables for the output
        self._best_obj  = 1000
        self._best_model= None
        self._best_disc = None 

        return
    
    def set_target(self, size):
        #set the intermediate size to minimze distance for. apply the settings

        self._target_size = size
        self._apply_target_settings()
        return
    
    def _apply_target_settings(self):
        #compute the exact solution for the given target, and the corresponding indices
        #in the Markov state model

        #get the "exact" solution from sampling data
        S = ClusterSizeData(self._traj_folder, recompute=False)
        dists = S.get_normalized_distribution(mass_weighted=True)
        self._exact = dists[:,self._target_size-1]

        #get the length of sampling data to know how many lags to go for
        self._num_lags = len(self._exact)

        #get the indices of the correspondind markov states
        self._target_indices = self._MM.filter_by_size(self._target_size)

        return
    
    def _solve_model(self, model):
        #solve the FKE for the current model 

        final_time   = round(self._num_lags/self._lag)
        soln         = model.get_FKE(T=final_time, p0="monomer_start")
        target_prob  = soln[:,self._target_indices].sum(1)

        return target_prob

    def _eval_obj(self, model):
        #evaluate the objective fn, norm of relative error b/w FKE solve and sampling 
        # average
        
        target_prob = self._solve_model(model)

        rel_error = np.zeros(self._num_lags, dtype=float)
        for i in range(self._num_lags):

            abs_error = target_prob[i]-self._exact[i]
            if self._exact[i] > 1e-5:
                rel_error[i] = (abs_error/self._exact[i])
            else:
                rel_error[i] = (abs_error)

        return np.linalg.norm(rel_error,2)
    
    def _plot_optimal_MSM_model(self):
        #plot the soln to FKE with best bin model and compare to sampling

        #solve FKE with best model
        target_prob = self._solve_model(self._best_model)

        #get the time discretizations
        t1 = range(len(target_prob))
        t2 = range(len(self._exact))

        plt.plot(t1, target_prob, t2, self._exact)
        plt.xlabel("Number of lags")
        plt.ylabel("Target Probability")
        plt.show()

        return
    

class BinOptimizerMC(BinOptimizer):

    def __init__(self, num_bins, lag, MM, traj_folder, initial_guess = None,
                 fixed_indices = None,
                 max_iter = 100, beta = 1.0, step_size = 0.05):
        
        #call the parent init
        super().__init__(num_bins, lag, MM, traj_folder, initial_guess, fixed_indices)

        #variables for the MC optimization
        self._max_iter = max_iter
        self._beta     = beta
        self._step_size= step_size

        #variables for output
        self._obj_series= np.zeros(max_iter, dtype=float)

        return
    
    def run_optimization(self, verbose = False):
        #perform the MC optimization

        #start by constructing the initial model and getting an energy
        model_builder = MultiMSMBuilder(self._initial_guess, self._MM, 
                                        self._traj_folder, self._lag)
        model0 = model_builder.make()
        E0 = self._eval_obj(model0)

        current_guess = self._initial_guess
        
        #perform Monte Carlo iterations to find a minimum
        for iter in range(self._max_iter):

            #perturb the guess
            new_guess = self._propose_move(current_guess)

            #make a new model and get its energy
            new_model = model_builder.remake(new_guess)
            new_energy= self._eval_obj(new_model)

            #accept or reject the new model
            accept = self._check_accept(new_energy-E0)
            if accept:
                E0 = new_energy
                current_guess = new_guess

            #keep track of the lowest seen energy and corresponding model
            if E0 < self._best_obj:
                self._best_obj  = E0
                self._best_disc = current_guess
                self._best_model= new_model

            #store the current value of E0 for plotting
            self._obj_series[iter] = E0

            #plot iteration updates
            if verbose:
                print("Iteration {}. Obj Fn Value {}.".format(iter, E0))

        #output some useful data about the optimal bins
        print("Best bin placements: ", self._best_disc.get_cutoffs())
        self._plot_series()
        self._plot_optimal_MSM_model()

        return
    
    def _propose_move(self, current_guess):
        #proposal move for MC search. randomly shift a random cutoff

        #make a new Disc object and copy the current
        proposal = Discretization(current_guess.get_cutoffs())

        #choose a bin index to move - b/w 1 and num_bins-1 inclusive
        index = self._choose_random_integer(1, self._num_bins-1)

        #choose an amount to move it - range [-step_size,step_size]
        random_shift = (2 * np.random.random() - 1) * self._step_size

        #apply the perturbation
        proposal.perturb(index, random_shift)

        return proposal

    def _choose_random_integer(self, lower, upper):
        #choose a random integer between lower and upper
        #check that it is not forbidden by the fixed list

        max_tries = 500

        for i in range(max_tries):

            r_int = np.random.randint(lower, upper+1)
            if r_int not in self._fixed_indices:
                return r_int
            
        #print error message if we cannot choose an index
        err_msg = "Could not choose a valid random cutoff in 500 tries. "
        err_msg+= "Are there enough degrees of freedom compared to fixed values?"
        raise RuntimeError(err_msg)
        return
    
    def _check_accept(self, delta_energy):

        u = np.random.rand()
        acc_prob = np.exp(-self._beta * delta_energy)
        print(acc_prob)
        if u < min(1.0, acc_prob):
            return True
        
        return False
    
    def _plot_series(self):
        #plot the time series of objective function values

        plt.plot(range(self._max_iter), self._obj_series)
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        plt.show()

        return

    
