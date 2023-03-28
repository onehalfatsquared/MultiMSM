import sys
import os
import fnmatch

import pickle

import numpy as np
import scipy
from scipy import sparse
from sklearn.preprocessing import normalize

from collections import defaultdict

from .MSM import MSM

#append parent directory to import util
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from ..util.discretization import Discretization
from ..util.state import MacrostateMap
from ..util.state import Monomer
from ..util.state import State

sys.path.pop(0)



class Collection:

    def __init__(self, discretization, macrostate_map, parameters = {}, lag = 1):

        #store the discretization and get number of discretized elements
        if not isinstance(discretization, Discretization):
            raise TypeError("Please provide a Discretization instance to Collection input 0")
        self.__discretization = discretization
        self.__num_elements   = discretization.get_num_intervals()

        #store the macrostate mapping and the total number of states
        if not isinstance(macrostate_map, MacrostateMap):
            raise TypeError("Please provide a MacrostateMap instance to Collection input 1")
        self.__macrostate_map = macrostate_map
        self.__num_states     = macrostate_map.get_num_states()

        #get the state and index for the Monomer (from the singleton class)
        self.__monomer_state  = Monomer().get_state()
        self.__monomer_index  = Monomer().get_index()

        #store the parameters for this MSM - key-value pairs in dict
        self.__parameters     = parameters
        self.__lag            = lag

        #create dict to associate each element to an MSM. init with empty MSMs
        self.__MSM_map        = dict()
        for i in range(self.__num_elements):

            self.__MSM_map[i+1] = MSM(self.__num_states, lag=lag)

        #init storage for solutions to forward and backward equations
        #size of these arrays depend on final time, a runtime variable
        self.__fke_soln = None
        self.__bke_soln = None
        self.__msm_indices = None

        #flag to mark when counts are done being added and have been normalized
        self.__counts_finalized = False

        #init count caching variables
        self.__count_cache = defaultdict(defaultdict(int).copy)
        self.__clear_cache_flag = False

        return

    def __load_from_cache(self, cluster_file):
        '''
        check if the cache file exists for this file, and if so, load the
        cache and add the counts from it. 

        returns the success status of loading from cache.
        '''

        #check for cache file existence
        cache_file = cluster_file.split(".cl")[0] + ".cache"
        if os.path.isfile(cache_file):

            #unpickle it and set the cache for self
            with open(cache_file,'rb') as infile:
                save_data = pickle.load(infile)
                self.__count_cache = save_data[0]
                old_states = save_data[1]

                #check that the maps are consistent via number of states
                if (old_states != self.__num_states):
                    err_msg = "MacrostateMaps must be consistent when loading from cache. "
                    err_msg+= "Call clear_cache() before processing to use new map. "
                    raise RuntimeError(err_msg)
                
            #add the counts from the cache to proper cont matrices
            self.__set_from_cache()
            return True

        return False

    def __save_cache(self, cluster_file):
        '''
        Save the cache via pickle. Include number of states in the macrostate map
        for easy comparison across runs
        '''

        #set cache file name and check for overwrites
        cache_file = cluster_file.split(".cl")[0] + ".cache"
        if os.path.isfile(cache_file) and not self.__clear_cache_flag:
            err_msg = "Saving this cache would overwrite an old one. This should not be possible. \
                       Delete the old file if this is desired."
            raise RuntimeError(err_msg)

        #save the cache as well as the number of states
        with open(cache_file,'wb') as outfile:
            pickle.dump((self.__count_cache, self.__num_states), outfile)
            print("Count cache saved to {}".format(cache_file))

        return


    def process_cluster_file(self, cluster_file, cache = False):
        '''
        Extract each transition from the given cluster file and add each count
        to the count matrix. 

        If cache is True, then the events will also be saved in self.__count_cache. 
        It is a dict that holds tuples of form (start_index,end_index), and associates
        a distribution of counts over each monomer fraction
        '''

        #try to load the cache file before constructing from scratch
        if (cache):

            #try load and return if succesful
            if not self.__clear_cache_flag:
                cache_status = self.__load_from_cache(cluster_file)
                if (cache_status):
                    return

            #if failed, clear the cache store a new one
            self.__reset_cache()

        #Process the cluster from scratch
        self.__process_transitions(cluster_file, cache=cache)

        #save the cached results if requested
        if (cache):
            self.__save_cache(cluster_file)

        return

    def __process_transitions(self, cluster_file, cache = False):
        '''
        Unpickle the cluster file and add up the counts
        '''

        #open the pickled file
        with open(cluster_file, 'rb') as f:
            sim_results = pickle.load(f)

        #extract the info from the pickle
        cluster_info = sim_results.cluster_info
        mon_fracs    = sim_results.monomer_frac
        mon_ids      = sim_results.monomer_ids

        #count all of the transitions that occur and add to collection
        self.__add_cluster_transitions(cluster_info, cache=cache)
        self.__add_monomer_monomer_transitions(mon_fracs, mon_ids, cache=cache)

        return


    def __add_transition(self, start_state, end_state, monomer_fraction, cache = False):
        #pick the correct MSM to update based on monomer frac, add a count

        #get the dictionary index of the MSM corresponding to this mon_frac
        msm_index = self.get_msm_index(monomer_fraction)

        #convert states to indices
        start_index = self.__macrostate_map.state_to_index(start_state)
        end_index   = self.__macrostate_map.state_to_index(end_state)

        #TODO - either test the efficiency of this, or make it so that states always in map
        if start_index is None or end_index is None:
            return

        #add a count to the appropriate MSM
        self.__add_count(start_index, end_index, msm_index)
        if (cache):
            self.__count_cache[(start_index,end_index)][int(monomer_fraction*100)] += 1

        return

    def __add_cluster_transitions(self, cluster_info, cache = False):
        '''
        loop over all transitions in cluster info and add them to the correct MSM in 
        collection. 
        '''

        lag = self.get_lag()

        #loop over each trajectory in cluster info
        for traj_num in range(len(cluster_info)):

            traj = cluster_info[traj_num]

            #get the length of the trajectory
            L = len(traj.get_data())

            #loop over all entries in the path, get transition for i+lag
            for start in range(L-lag+1):

                #get list of all transitions within the lag interval, and the monomer fraction at that time
                transitions = traj.get_transitions(start, lag)
                mon_frac    = traj.get_data()[start]['monomer_fraction']

                #add them one by one to the dict
                for transition in transitions:
                    state1   = transition[0]
                    state2   = transition[1]
                    self.__add_transition(state1, state2, mon_frac, cache=cache)

        return 

    def __add_monomer_monomer_transitions(self, mon_fracs, mon_ids, cache = False):
        '''
        Loop over each [frame, frame+lag] of the simulation to determine how many 
        monomer -> monomer transitions there are, and at what monomer fraction
        '''

        #init dict to store number of counts in each discretization window
        MMcounts = defaultdict(int)
        lag      = self.get_lag()

        #loop over all lags in the trajectory
        L = len(mon_ids)
        for i in range(L-lag):

            #get the monomer fraction at this frame, and the corresponding discretization index
            mon_frac  = mon_fracs[i]
            msm_index = self.get_msm_index(mon_frac) 

            #get the set of monomer ids at each frame
            set1 = mon_ids[i]
            set2 = mon_ids[i+lag]

            #update the count with the number of monomers that persisted
            num_persisted = len(set1.intersection(set2))
            MMcounts[msm_index] += num_persisted
            if (cache):
                self.__count_cache[(self.__monomer_index, self.__monomer_index)][int(mon_frac*100)] += num_persisted

        #update the MSMs in the collection with the finalized counts
        self.__update_monomer_monomer_counts(MMcounts)
        return

    def __add_count(self, start_index, end_index, msm_index, counts = 1):

        self.__MSM_map[msm_index].add_count(start_index, end_index, counts = counts)
        return

    def __update_monomer_monomer_counts(self, MMcounts):
        #update the (monomer,monomer) entry of count matrix in each MSM according to value in MMcount

        #get the monomer state index (should be 0)
        monomer_index = self.__monomer_index

        #loop over each key (msm index) and append number of mon -> mon counts
        for key in MMcounts:
        
            #check if the key is a non-allowed index (i.e. mon frac of exactly 0 or 1)
            if (key == 0):
                msm_key = 1 #map this entry to the first interval
            elif (key == (self.__num_elements+1)):
               msm_key = key-1 #map entry to the last interval
            else:
                msm_key = key
            
            #extract and add the counts
            num_counts = MMcounts[key]
            self.__add_count(monomer_index, monomer_index, msm_key, counts = num_counts)

        return

    def __set_from_cache(self):
        '''
        Use the transition data stored in the cache to add to the count matrix. 
        '''

        #outermost dict has keys (start_index,end_index) and values are dicts
        for transition in self.__count_cache:

            #extract the indices and the inner dict
            state1 = transition[0]
            state2 = transition[1]
            data = self.__count_cache[transition]

            #innermost dict has keys (100*mon_frac) and values are counts
            for frac in data:

                #get counts, compute an index from frac, add the counts
                counts = data[frac]
                msm_index = self.get_msm_index(self.__fix_zero_one(frac/100))
                self.__add_count(state1, state2, msm_index, counts = counts)

        return


    def finalize_counts(self):
        #call finalize counts on each MSM. Constructs a transition matrix from the counts added so far

        for i in range(self.__num_elements):

            self.__MSM_map[i+1].finalize_counts(self.__macrostate_map)


        #set a flag that there are now transition matrices that can be used
        self.__counts_finalized = True

        return

    def get_lag(self):

        return self.__lag

    def get_msm_index(self, monomer_fraction):

        return self.__discretization.determine_interval(monomer_fraction)

    def get_msm(self, msm_index):

        return self.__MSM_map[msm_index]

    def get_transition_matrix(self, msm_index):

        return self.get_msm(msm_index).get_transition_matrix()

    def get_count_matrix(self, msm_index):

        return self.get_msm(msm_index).get_count_matrix()

    def get_FKE(self, p0 = None, T = 100):

        #check if a solution to the FKE has been computed and return it
        if self.__fke_soln is not None:
            return self.__fke_soln

        #check if a solution can be computed and do so with given or default values
        if not self.__counts_finalized:

            err_msg =  "Transition matrices have not been finalized. Solution to FKE "
            err_msg += "cannot be computed yet"
            raise RuntimeError(err_msg)

        #check for distribution and set default if none
        if p0 is None:
            init_dist = np.zeros(self.__num_states, dtype=float)
            init_dist[self.__monomer_index] = 1.0
            print("Warning: Initial distribution not specified. Defaulting to 100% monomer")
        else:
            init_dist = p0

        #solve FKE and return solution
        print("Solving FKE with supplied initial distribution and T={} lags.".format(T)) 
        self.solve_FKE(init_dist, T)
        return self.__fke_soln


    def print_all_transition_matrices(self):

        for i in range(self.__num_elements):

            m = self.__MSM_map[i+1]
            print("For m_frac in {}".format(self.__discretization.interval_index_to_bounds(i+1)))
            print(m.get_transition_matrix())

    def print_all_count_matrices(self):

        for i in range(self.__num_elements):

            m = self.__MSM_map[i+1]
            print("For m_frac in {}".format(self.__discretization.interval_index_to_bounds(i+1)))
            print(m.get_count_matrix())


    def solve_FKE(self, p0, T):
        '''
        Solve the Forward Kolmogorov Equation for the probability of states as a function
        of time. 
        p0 is the initial distribution, row vector (nparray)
        T is the final time (int # of lag times)

        In solving this equation, the transition matrix that will be used each step depends on the 
        monomer fraction. We will track the msm index of the appropriate transition matrix and 
        save the time series. This will be needed to solve the backward equations. 
        '''

        #first check the distribution is an nparray with the correct dimensions
        if type(p0) is not np.ndarray:
            p0 = np.array(p0, dtype=float)

        if len(p0) != self.__num_states:
            err_msg =  "The length of the supplied initial distribution ({})".format(len(p0))
            err_msg += " does not match the number of states ({})".format(self.__num_states)
            raise ValueError(err_msg)

        #init an array to store the time dependent solution, as well as MSM indices
        p          = np.zeros((T+1, self.__num_states), dtype=float)
        indices    = np.zeros(T+1, dtype=int)
        p[0, :]    = p0
        mon_frac0  = p0[self.__monomer_index]
        indices[0] = self.get_msm_index(self.__fix_zero_one(mon_frac0))

        #solve the FKE, grabbing the relevant transition matrix each iteration
        for t in range(T):

            #get the transition matrix for the current time step
            TM = self.get_transition_matrix(indices[t])

            #update the probabilities 1 step in future
            p[t+1, :] = p[t, :] * TM

            #get the index for the next transition matrix from monomer frac
            current_mon_frac = p[t+1,self.__monomer_index]
            indices[t+1] = self.get_msm_index(self.__fix_zero_one(current_mon_frac))

        #store the solution and indices
        self.__fke_soln    = p
        self.__msm_indices = indices

        #return the solution as well as storing? maybe a copy?
        return

    def solve_BKE(self, fN, T):
        '''
        Solve the Backward Kolmogorov Equation for the probability of states as a function
        of time. 
        fN is the indicator (column) vector for a target set at final time (nparray), 
        T is the final time (int # of lag times)

        In solving this equation, the transition matrix that will be used each step depends on the 
        monomer fraction. The time sequence is stored in the call to solve_FKE and accessed here. 
        '''

        #first check that the forward equation has been solved. throw error if not
        if self.__msm_indices is None:
            err_msg  = "The sequence of transition matrices has not been computed."
            err_msg += " Please call solve_FKE() before calling solve_BKE()"
            raise RuntimeError(err_msg)

        #check if the fN distribution is an nparray with the correct dimensions
        if type(fN) is not np.ndarray:
            fN = np.array(fN, dtype=float)

        if len(fN) != self.__num_states:
            err_msg =  "The length of the supplied initial distribution ({})".format(len(fN))
            err_msg += " does not match the number of states ({})".format(self.__num_states)
            raise ValueError(err_msg)

        #init storage for the solution
        f      = np.zeros((self.__num_states, T+1), dtype=float)
        f[:,T] = fN

        #solve the BKE
        for t in range(T):

            #get the appropriate transition matrix and perform a step
            TM         = self.get_transition_matrix(self.__msm_indices[T-1-i])
            f[:,T-1-i] = TM * F[:,T-i]

        #store the calculated soln
        self.__bke_soln = f

        #return the solution? 
        return

    def __fix_zero_one(self, mon_frac):
        #if the monomer fraction is exactly zero or 1, give it a slight push

        if mon_frac > (1-1e-6):
            mon_frac -= 1e-6

        elif mon_frac < 1e-6:
            mon_frac += 1e-6

        return mon_frac

    def clear_cache(self):
        '''
        Calling this method will set a flag to avoid the check for inconsistent maps. 
        The next time files are processed, a new cache will be built and saved.
        '''

        self.__clear_cache_flag = True
        return

    def __reset_cache(self):

        self.__count_cache = defaultdict(defaultdict(int).copy)
        return

    def get_effective_MSM(self):

        num_states = self.__num_states
        count_matrix = scipy.sparse.dok_matrix((num_states, num_states), dtype=int)

        for key in self.__MSM_map:

            count_matrix += self.get_msm(key).get_count_matrix()

        msm = MSM(num_states, lag=self.__lag)
        msm.set_count_matrix(count_matrix, self.__macrostate_map)

        return msm
