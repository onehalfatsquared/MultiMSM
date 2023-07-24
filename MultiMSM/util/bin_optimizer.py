'''
Implements an optimizer to choose the "best" discretization for a MultiMSM according to 
a user provided metric. Starting choices are to minimze the distance between a given 
structure yield curve, and the sampling approximation to that curve. Performs a Monte 
Carlo search over bin placements at fixed number of bins. 
'''

import numpy as np

from MultiMSM.MSM.MultiMSM import MultiMSMBuilder
from MultiMSM.MSM.MultiMSM import MultiMSMSolver
from MultiMSM.util.sampling_data import ClusterSizeData
from MultiMSM.util.sampling_data import MicrostateCollectionData
from MultiMSM.util.discretization import Discretization

from MultiMSM.util.state import TargetStates

import matplotlib.pyplot as plt



class BinOptimizer:

    def __init__(self, num_bins, lag, MM, traj_folder, final_time = None,
                 initial_guess = None, fixed_indices = None, obj_norm = 2, 
                 target_size = 1, target_states = None,
                 compare_long = False):

        #set user inputs
        self._num_bins    = num_bins
        self._lag         = lag
        self._MM          = MM
        self._traj_folder = traj_folder
        self._samples_loc = traj_folder
        self._obj_norm    = obj_norm
        self._final_time  = final_time

        #if the comparison is over the long trajectories, overwrite sample location
        self._compare_long = compare_long
        if compare_long:
            self._samples_loc += "long/"

        #check for initial guess and right type
        if initial_guess is not None:
            if isinstance(initial_guess, Discretization):
                self._initial_guess = initial_guess
                self._fixed_indices = []
                self._num_bins = initial_guess.get_num_intervals()
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

        #set variables for the target state, create comparison results, setup solver
        self._target_size = None
        self._manage_target_settings(target_size, target_states)
        self._setup_solve()

        #variables for the output
        self._best_obj  = 1000
        self._best_model= None
        self._best_disc = None 

        return
    
    def _manage_target_settings(self, target_size, target_states):
        #set the target state settings via the supplied input

        #if target states is None, use size. Otherwise, use states
        if target_states is None:
            self.set_target_size(target_size)
            return

        #if we reach here, determine what was given as input for target_states
        if isinstance(target_states, list):
            target_indices = target_states
        elif isinstance(target_states, TargetStates):
            target_indices = target_states.get_indices()
        else:
            err_msg = "Please provide either a list of indices for target_states "
            err_msg+= "or a TargetStates object."

        #set the target using these indices
        self.set_target_indices(target_indices)
        return
    
    def set_target_size(self, size):
        #set the intermediate size to minimze distance for. apply the settings

        self._target_size = size
        self._apply_size_settings()
        return
    
    def set_target_indices(self, indices):

        self._target_indices = indices
        self._apply_target_settings()
        return
    
    def _apply_size_settings(self):
        #make a ClusterSizeData object to extract the 'exact' solution for the
        #given cluster size 

        #get the "exact" solution from sampling data
        S = ClusterSizeData(self._samples_loc, recompute=False)
        dists = S.get_normalized_distribution(mass_weighted=True)
        self._exact = dists[:,self._target_size-1]

        #get the length of sampling data to know how many lags to go for
        self._num_lags = len(self._exact)

        #get the indices of the correspondind markov states
        self._target_indices = self._MM.filter_by_size(self._target_size)

        return
    
    def _apply_target_settings(self):
        #make a MicrostateCollectionData object to extract the 'exact' solution for 
        #the summed yield of all state indices provided

        #convert indices into actual states
        target_states = [self._MM.index_to_state(i) for i in self._target_indices]

        #get the data for the desired states and make a time series of yield
        MS = MicrostateCollectionData(self._samples_loc, target_states)
        self._exact = MS.get_time_series()

        #get the length of sampling data to know how many lags to go for
        self._num_lags = len(self._exact)

        return
    
    def _setup_solve(self):
        #setup all variables needed to do a FKE solve from the MSM

        #set final time based on user inputs
        if self._final_time is None: 
            #use full length of the specified data
            self._final_time = self._num_lags
        else:
            #check if the user given final time is longer than the data
            self._final_time = int(self._final_time)
            if self._final_time > self._num_lags:

                #give a warning that the specified time is too long
                warn_msg = "Warning: specified final time {} is greater than the final ".format(self._final_time)
                warn_msg+= "time from sampling {}, using files from {}.\n".format(self._num_lags, self._samples_loc)
                warn_msg+= "Continuing with final time {}".format(self._num_lags)
                print(warn_msg)

                #cap out the time at the full sampling time
                self._final_time = self._num_lags
        
        #set the actual final time in sim units instead of lags
        self._Tf   = self._final_time / self._lag

        #set all monomer initial condition
        self._msIC = "monomer_start"

        return
    
    def _solve_model(self, model):
        #solve the FKE for the current model 

        #create the solver and solve the FKE with given conditions
        solver      = MultiMSMSolver(model)
        target_prob = solver.get_FKE(self._final_time, p0=self._msIC, frac=0.25, 
                                     target_index_list = self._target_indices)
        return target_prob

    def _eval_obj(self, model):
        #evaluate the objective fn, norm of relative error b/w FKE solve and sampling 
        # average
        
        #solve the current model
        target_prob = self._solve_model(model)

        #init and compute the relative error between MSM and sampling
        rel_error = np.zeros(self._final_time, dtype=float)
        for i in range(self._final_time):

            #get absolute error
            abs_error = target_prob[i]-self._exact[i]

            #scale to get rel. error. use absolute if true solution is very small
            if self._exact[i] > 1e-5:
                rel_error[i] = (abs_error/self._exact[i])
            else:
                rel_error[i] = (abs_error)

        return np.linalg.norm(rel_error, self._obj_norm)
    
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
    

class BinOptimizerSequential(BinOptimizer):
    '''
    Performs bin optimization by iterating over the dividier positions one at a 
    time, and choosing the optimal placement of that divider with the others held
    fixed, by sampling values between the surrounding dividers. 

    Can set the number of sweeps, defined as optimizing each bin once sequentially. 
    Can set the number of points to sample in each interval. We force this number to 
    be even, else the midpoint would re-sample the same configuration. 
    '''

    def __init__(self, num_bins, lag, MM, traj_folder, final_time = None,
                 initial_guess = None, fixed_indices = None, num_files = None,
                 obj_norm = 2, target_size = 1, target_states = None,
                 compare_long = False, 
                 num_sweeps = 1, samples_per_div = 4):
        
        #call the parent init
        super().__init__(num_bins, lag, MM, traj_folder, final_time, initial_guess, 
                         fixed_indices, obj_norm, target_size, target_states, 
                         compare_long)

        #variables for the optimization
        self._samples_per_div = samples_per_div + (samples_per_div%2)
        self._num_sweeps      = num_sweeps
        
        #derived variables
        self._num_dividers = self._num_bins - 1

        #construct the model builder object and create the initial model
        self._model_builder = MultiMSMBuilder(self._initial_guess, self._MM, 
                                              self._traj_folder, self._lag,
                                              num_files=num_files)
        self._model0 = self._model_builder.make()
        self._obj0   = self._eval_obj(self._model0)

        self._current_guess = self._initial_guess

        return
    
    def run_optimization(self, verbose = False):
        #perform sweeps over all of the dividers sequentially
        
        for sweep in range(self._num_sweeps):
            
            #init a flag to check if this sweep actually changed anything
            self._modification_flag = False

            #loop over each divider
            for divider in range(self._num_dividers):

                #get equally spaced test points to potentially replace this divider
                test_points = self._get_test_points(divider)

                #sample each point and see if it improves the objective function
                for point in test_points:

                    self._test_new_divider(divider, point, verbose)
                    
            #print information about the last sweep
            if verbose:
                print("Sweep #{}, current obj fn {}.".format(sweep+1, self._obj0))

            #if no sweep was made this update, exit sweep loop
            if not self._modification_flag:
                break

        #set current guess as best in case where the placements can't be improved
        if self._best_disc is None:
            self._best_disc  = self._current_guess
            self._best_model = self._model0

        #print info on overall optimization 
        print("Best bin placements", self._best_disc.get_cutoffs())
        self._plot_optimal_MSM_model()

        return
    
    def _get_test_points(self, divider):
        #use the current cutoffs to generate uniformly separated points between
        #the current divider's neighbors
        #start with the rightmost interior divider and work left

        #get the current values of the cutoffs
        cutoffs = self._current_guess.get_cutoffs()

        #get the location of the left and right neighboring dividers
        right_div = cutoffs[-1-divider]
        left_div  = cutoffs[-1-divider-2]

        #place samples_per_div points between these cutoffs to sample
        test_points = np.linspace(left_div, right_div, self._samples_per_div+2)
        test_points = test_points[1:(1+self._samples_per_div)]

        return test_points
    
    def _test_new_divider(self, divider, point, verbose):
        #see if the new divider location, point, improves the objective function
        #if so, update the variables keeping track of the best

        #create new discretization with these cutoffs
        new_cutoffs = self._current_guess.get_cutoffs().copy()
        new_cutoffs[-1-divider-1] = point
        new_guess = Discretization(new_cutoffs)

        #make and evaluate a new model
        new_model = self._model_builder.remake(new_guess)
        new_obj   = self._eval_obj(new_model)

        #check if new model improves upon old
        if new_obj < self._obj0:

            if verbose:
                print("Divider #{} changed to {}. Old obj {}, new obj {}.".format(divider, point, self._obj0, new_obj))

            #overwrite the previous guess 
            self._obj0 = new_obj
            self._current_guess = new_guess

            #keep track of best objfn, discretization, and model
            self._best_obj  = self._obj0
            self._best_disc = self._current_guess
            self._best_model= new_model

            #update the modification flag to denote a change was made to the discr.
            self._modification_flag = True

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

    

