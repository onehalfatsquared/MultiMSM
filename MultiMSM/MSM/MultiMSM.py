import sys
import os
import fnmatch

import pickle

import numpy as np
import scipy
from scipy import sparse

from collections import defaultdict, Counter

import time
import random

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


class MultiMSMBuilder:
    '''
    This class is responsible for actually constructing and returning the MultiMSM. 
    It takes in the discretization, macrostate mapping, and the location of the 
    .cl file trajectories to fill a Collection object. 

    Can also cache the observation counts as a function of the monomer fraction to
    allow for quick reconstruction of the MultiMSM with a different discretization. 
    The number of files to include can also be set to study convergence with regard to 
    number of samples. Files are sorted by file name and will simply leave off the 
    trailing N-num_files files.  
    '''

    def __init__(self, discretization, macrostate_map, traj_folder, lag = 1,
                 params = {}, cache = True, clear_cache = False, num_files = None,
                 prune_tol = None, verbose = False, bootstrap = False):
        
        #store any optional input parameters
        self.__clear_cache = clear_cache
        self.__cache       = cache
        self.__num_files   = num_files
        self.__verbose     = verbose
        self.__params      = params
        self.__lag         = lag
        self.__prune_tol   = prune_tol
        self.__bootstrap   = bootstrap

        #make an empty collection to form the MultiMSM, store the discretization and map
        self.C = Collection(discretization, macrostate_map, parameters=params, lag=lag,
                            prune_tol=prune_tol)
        self.__discretization = discretization
        self.__macrostate_map = macrostate_map

        #gather the relevant cl files to construct the MultiMSM, all of them
        self.__gather_files(traj_folder)

        #prepare a dict to store num_files for each run type
        self.__prepare_num_files(num_files)

        #clear cache if requested
        if (cache and clear_cache):
            self.C.clear_cache()

        return 
    
    def make(self):
        #process all the files to construct the MultiMSM and return it

        if not self.C.is_finalized():
            self.__process_files(self.__cache, self.__verbose)

        return self.C
    
    def remake(self, new_discretization, new_lag = None):
        #remake the MultiMSM using a new discretization and potentially new lag

        self.__discretization = new_discretization
        if new_lag is not None:
            self.__lag = new_lag

        self.C = Collection(self.__discretization, self.__macrostate_map, 
                            parameters=self.__params, lag=self.__lag)
        
        self.__process_files(self.__cache, self.__verbose)

        return self.C


    def __gather_files(self, traj_folder):
        #walk through subdirectories of the given folder to find all .cl files

        #check that the supplied traj folder exists
        if (not os.path.exists(traj_folder)):
            raise("Specified folder could not be found")

        #init storage for file paths of different run types
        self.__base_paths  = []
        self.__short_paths = []
        self.__cont_paths  = []
        self.__long_paths  = []

        #get all the base paths
        self.__base_paths = self.__walk_directory(traj_folder)

        #check for existence of other folders and gather from them
        if (os.path.exists(traj_folder+"short/")):
            self.__short_paths = self.__walk_directory(traj_folder+"short/")

        if (os.path.exists(traj_folder+"continue/")):
            self.__cont_paths = self.__walk_directory(traj_folder+"continue/")

        if (os.path.exists(traj_folder+"long/")):
            self.__long_paths = self.__walk_directory(traj_folder+"long/")

        self.__path_map = {"base":self.__base_paths, "short":self.__short_paths,
                           "cont":self.__cont_paths, "long": self.__long_paths}

        #handle frac and reduce paths separately, since there are distributions here
        self.__frac_exists, self.__frac_paths, self.__frac_counts = self.__gather_extra_paths(traj_folder, "frac")
        self.__red_exists, self.__red_paths, self.__red_counts = self.__gather_extra_paths(traj_folder, "reduced")
        
        return
    
    def __gather_frac_paths(self, traj_folder):
        #determine if there is a 'frac' folder, and populate the dict with its paths

        #if no frac folder, just exit this function, set existence flag to false
        if not (os.path.exists(traj_folder+"frac/")):
            self.__frac_exists = False
            return
        
        #set existence flag to true
        self.__frac_exists = True

        #init dicts to store paths and counts per frac interval
        self.__frac_paths  = dict()
        self.__frac_counts = dict()

        #walk the directory to get all paths
        all_frac_paths = self.__walk_directory(traj_folder+"frac/")

        #separate the paths by frac value
        for i in range(10):
            self.__frac_paths[i] = [path for path in all_frac_paths if "frac{}".format(i) in path]  
            self.__frac_counts[i]= len(self.__frac_paths[i])

        return
    
    def __gather_extra_paths(self, traj_folder, path_type):
        #determine if there is a 'frac' folder, and populate the dict with its paths

        #if no frac folder, just exit this function, set existence flag to false
        if not (os.path.exists(traj_folder+path_type)):
            return False, dict(), {i:0 for i in range(10)}

        #init dicts to store paths and counts per frac interval
        paths  = dict()
        counts = dict()

        #walk the directory to get all paths
        all_paths = self.__walk_directory(traj_folder+path_type+"/")

        #separate the paths by frac value
        for i in range(10):
            paths[i] = [path for path in all_paths if path_type+str(i) in path]  
            counts[i]= len(paths[i])

        return True, paths, counts
    

    def __walk_directory(self, dir_path):
        #walk through each subdirectory of the given path and accumulate .cl files

        #init list for storing paths
        file_paths = []

        #get all the valid 1 level deep subdirectories and loop over them
        subdirs = [x for x in os.listdir(dir_path) if os.path.isdir(dir_path+x)]
        for subdir in subdirs:

            #check files in the subdirectory for the .cl extension, add those
            for file in os.listdir(dir_path+subdir):
                
                if file.endswith('.cl'):
                    file_paths.append(os.path.join(dir_path+subdir, file))


        #sort the list of filenames and return them
        file_paths.sort()

        return file_paths


    def __prepare_num_files(self, num_files):
        #tell the constructor how many of each type of file to use
        #accepts None, an integer (# base paths), or a dict

        self.__process_frac = False
        self.__process_red  = False

        self.__file_counter = defaultdict(int)
        self.__frac_counter = {}
        self.__red_counter  = {}

        #case 1 - None -> use all files
        if num_files is None:

            for path_type in self.__path_map.keys():
                self.__file_counter[path_type] = len(self.__path_map[path_type])

            return

        #case 2 - int -> use specified number of base files, none of the others
        if isinstance(num_files, int):

            for path_type in self.__path_map.keys():
                self.__file_counter[path_type] = 0

            self.__file_counter['base']  = min(len(self.__base_paths), num_files)

            return

        #case 3 - dict -> copy values, ensuring they are valid
        if isinstance(num_files, dict):

            #init all types to 0
            self.__frac_counter = {i:0 for i in range(10)}
            self.__red_counter = {i:0 for i in range(10)}
            for path_type in self.__path_map.keys():
                self.__file_counter[path_type] = 0

            #add the user specified counts for each path type
            for k,v in num_files.items():
                if k != "frac" and k != 'reduced':
                    self.__file_counter[k] = min(v, len(self.__path_map[k]))

                #handle "frac" case separately
                elif k == "frac" and self.__frac_exists:

                    #set flag to process these trajectories
                    self.__process_frac = True

                    #check if we request 'all' or a subset
                    if v == "all":
                        self.__frac_counter = self.__frac_counts

                    #cjeck if there is a dict holding a distribution of files to use
                    elif isinstance(v, dict):
                        for fk,fv in v.items():
                            self.__frac_counter[fk] = min(fv, self.__frac_counts[fk])

                    #print an error message that the 'frac' value provided is invalid
                    else:
                        err_msg = "The supplied value for the 'frac' key is invalid. "
                        err_msg+= "It only supports 'all' or a dict of keys with "
                        err_msg+= "integers 0:9 and values the number of files to use."
                        raise RuntimeError(err_msg)
                
                #handle "frac" case separately
                elif k == "reduced" and self.__red_exists:

                    #set flag to process these trajectories
                    self.__process_red = True

                    #check if we request 'all' or a subset
                    if v == "all":
                        self.__red_counter = self.__red_counts

                    #cjeck if there is a dict holding a distribution of files to use
                    elif isinstance(v, dict):
                        for rk,rv in v.items():
                            self.__red_counter[rk] = min(rv, self.__red_counts[rk])

                    #print an error message that the 'red' value provided is invalid
                    else:
                        err_msg = "The supplied value for the 'reduced' key is invalid"
                        err_msg+= ". It only supports 'all' or a dict of keys with "
                        err_msg+= "integers 0:9 and values the number of files to use."
                        raise RuntimeError(err_msg)

            #check if they supplied an unsupported path type
            if len(self.__file_counter.keys()) > len(self.__path_map.keys()):
                err_msg = "The supplied file count dict has invalid keys. "
                err_msg+= "It only supports 'base', 'short', 'long', 'cont', 'frac', and 'reduced'."
                raise RuntimeError(err_msg)

            return

        #if we reach here, an invalid data type was passed for num_files
        err_msg = "Type {} for num_files is not supported. Please supply None, "
        err_msg+= "an int, or a dict with supported keys. "
        raise TypeError(err_msg)

        return
    
    def __process_files(self, cache, verbose):
        #loop over given files, adding counts to the collection. 

        if verbose:
            print("\nProcessing files for MSM construction...",end=' ')

        a = time.time()

        #do processing of each type of file
        for path_type in self.__path_map.keys():

            #get the paths to the trajectories of each type and the number to use
            paths     = self.__path_map[path_type]
            num_paths = self.__file_counter[path_type]

            self.__do_processing(paths, num_paths, cache)

        #process frac files separately if requested
        if self.__process_frac:
            for k,v in self.__frac_paths.items():

                #filter to the number of requested paths
                paths = v
                num_paths = self.__frac_counter[k]

                self.__do_processing(paths, num_paths, cache, kill_monomer=True)

        #process reduced files separately if requested
        if self.__process_red:
            for k,v in self.__red_paths.items():

                #filter to the number of requested paths
                paths = v
                num_paths = self.__red_counter[k]
                starting_frac = (k+1)/10

                self.__do_processing(paths, num_paths, cache, ratio=starting_frac)

        b = time.time()
            
        #finalize the counting and return
        self.C.finalize_counts()

        #check if there were any map misses and print warning if there were
        misses = self.C.get_misses()
        if misses > 0:
            w_msg = "Warning: there were {} transitions that could ".format(misses)
            w_msg+= "not be logged because the state was missing from the macrostate "
            w_msg+= "map. Ensure all trajectory databases of states have been combined."
            

        #print optional message detailing number of files used per type
        if verbose:
            num_paths = sum(self.__file_counter.values())
            print("Processing complete.")
            num_paths += self.__output_trajectory_details()
            print("Processing took {}s. Average {}s per file\n".format(b-a, (b-a)/num_paths))
        
        return

    def __do_processing(self, paths, counts, cache, ratio=1.0, kill_monomer=False):
        #actually do the file processing

        #make list containing all the paths to sample
        path_list = paths[0:counts]
        #if bootstrapping, fill with random samples from paths with replacement
        if self.__bootstrap:
            new_paths = random.choices(path_list, k=counts)
            path_list = new_paths

        #loop over requested number of files
        for next_file in path_list:

            #add all transitions from current file, cache if requested
            self.C.process_cluster_file(next_file, cache=cache, ratio=ratio, 
                                        kill_monomer=kill_monomer)

        return
    
    def __output_trajectory_details(self):
        #print a summary of how many trajectories were processed among each type

        #init number of frac trajectories processed
        total_frac = 0

        #print counts of each type of non-frac trajs
        print("Base trajectories processed: {}".format(self.__file_counter['base']))
        print("Short trajectories processed: {}".format(self.__file_counter['short']))
        print("Long trajectories processed: {}".format(self.__file_counter['long']))
        print("Disassembly trajectories processed: {}".format(self.__file_counter['dis']))
        print("Continued trajectories processed: {}".format(self.__file_counter['cont']))

        #process frac trajectories separately
        if self.__process_frac:
            total_frac = np.sum(list(self.__frac_counter.values()))
            print("\nTotal frac target trajectories processed: {}".format(total_frac))
            for k in self.__frac_counter:
                if self.__frac_counter[k] > 0:
                    print("Frac {}-{}: {}".format(k/10,k/10+.1,self.__frac_counter[k]))
            print()

        #process reduced trajectories separately
        if self.__process_red:
            total_frac = np.sum(list(self.__red_counter.values()))
            print("\nTotal reduced frac target trajectories processed: {}".format(total_frac))
            for k in self.__red_counter:
                if self.__red_counter[k] > 0:
                    print("Frac {}-{}: {}".format(k/10,k/10+.1,self.__red_counter[k]))
            print()

        #return number of frac trajs
        return total_frac





class Collection:
    '''
    The Collection class represents a MultiMSM, a collection of MSMs at different 
    values of the monomer fraction of a system. 

    Takes in a discretization of the interval [0,1] that represents all possible 
    monomer fractions, and a mapping from a State object to an integer index. 

    Contains methods for processing .cl files, which will add all observed transitions
    between the discrete states to the appropriate MSMs, depending on the value of the 
    monomer fraction for which the transition occurred. 

    The counts must be finalized in order to get a probability transition matrix
    representation of each MSM. Required for passing to KE solvers. 
    
    '''

    def __init__(self, discretization, macrostate_map, parameters = {}, lag = 1,
                 prune_tol = None, verbose = False):

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
        self.__prune_tol      = prune_tol
        self.__default_pruning= 1

        self.__verbose        = verbose

        #create dict to associate each element to an MSM. init with empty MSMs
        self.__MSM_map        = dict()
        for i in range(self.__num_elements):

            if isinstance(prune_tol, dict):
                try:
                    pruning = prune_tol[i+1]
                except:
                    pruning = self.__default_pruning
            else:
                pruning = prune_tol

            self.__MSM_map[i+1] = MSM(self.__num_states, lag=lag, 
                                      prune_threshold=pruning)

        #store number of observations of each monomer fraction
        self.__frac_freqs  = Counter()

        #flag to mark when counts are done being added and have been normalized
        self.__counts_finalized = False
        self.__clusterized = False

        #init count caching variables
        self.__count_cache = defaultdict(defaultdict(int).copy)
        self.__freq_cache  = defaultdict(int)
        self.__clear_cache_flag = False

        #keep log of how many transitions are from states not in the MM
        self.__map_misses = 0

        return
    
    def clear_cache(self):
        '''
        Calling this method will set a flag to avoid the check for inconsistent maps. 
        The next time files are processed, a new cache will be built and saved.
        '''

        self.__clear_cache_flag = True
        return

    def __reset_cache(self):

        self.__count_cache = defaultdict(defaultdict(int).copy)
        self.__freq_cache  = defaultdict(int)
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
                self.__freq_cache  = save_data[1]
                old_states         = save_data[2]

                #check that the maps are consistent via number of states
                if (old_states != self.__num_states):
                    err_msg = "MacrostateMaps must be consistent when loading from cache. "
                    err_msg+= "Call clear_cache() before processing to use new map. "
                    raise RuntimeError(err_msg)
                
            #add the counts from the cache to proper count matrices
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
            pickle.dump((self.__count_cache, self.__freq_cache, self.__num_states), outfile)
            print("Count cache saved to {}".format(cache_file))

        return


    def process_cluster_file(self, cluster_file, cache = False, ratio=1,
                             kill_monomer=False):
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

            #if failed, clear the cache, store a new one
            self.__reset_cache()

        #Process the cluster from scratch
        self.__process_transitions(cluster_file, cache=cache, ratio=ratio, 
                                   kill_monomer=kill_monomer)

        #save the cached results if requested
        if (cache):
            self.__save_cache(cluster_file)

        return

    def __process_transitions(self, cluster_file, cache = False, ratio=1,
                              kill_monomer=False):
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
        self.__add_cluster_transitions(cluster_info, cache=cache, ratio=ratio,
                                       kill_monomer=kill_monomer)
        if not kill_monomer:
            self.__add_monomer_monomer_transitions(mon_fracs, mon_ids, cache=cache, 
                                                   ratio=ratio)

        return


    def __add_transition(self, start_state, end_state, monomer_fraction, cache = False):
        #pick the correct MSM to update based on monomer frac, add a count

        #get the dictionary index of the MSM corresponding to this mon_frac
        msm_index = self.get_msm_index(monomer_fraction)

        #convert states to indices
        start_index = self.__macrostate_map.state_to_index(start_state)
        end_index   = self.__macrostate_map.state_to_index(end_state)

        #log if we find a state not in the map, skip this transition
        if start_index == -1 or end_index == -2:
            self.__map_misses += 1
            return
        

        #add a count to the appropriate MSM
        self.__add_count(start_index, end_index, msm_index)
        if (cache):
            self.__count_cache[(start_index,end_index)][int(monomer_fraction*100)] += 1

        return

    def __add_cluster_transitions(self, cluster_info, cache = False, ratio=1, 
                                  kill_monomer=False):
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
                mon_frac    = traj.get_data()[start]['monomer_fraction'] * ratio

                #add them one by one to the dict
                for transition in transitions:
                    state1   = transition[0]
                    #do not log monomer transitions if kill_monomer is true
                    if kill_monomer and state1.get_size() == 1:
                        continue #skips to next iteration in loop over start
                    state2   = transition[1]
                    self.__add_transition(state1, state2, mon_frac, cache=cache)

        return 

    def __add_monomer_monomer_transitions(self, mon_fracs, mon_ids, cache = False, 
                                          ratio=1):
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
            mon_frac  = mon_fracs[i] * ratio
            msm_index = self.get_msm_index(mon_frac) 

            #get the set of monomer ids at each frame
            set1 = mon_ids[i]
            set2 = mon_ids[i+lag]

            #update the count with the number of monomers that persisted
            num_persisted = len(set1.intersection(set2))
            MMcounts[msm_index] += num_persisted
            if (cache):
                self.__count_cache[(self.__monomer_index, self.__monomer_index)][int(mon_frac*100)] += num_persisted
                self.__freq_cache[int(mon_frac*100)] += 1

            #update number of times each monomer frac is seen
            self.__frac_freqs[int(mon_frac*100)] += 1

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
        Also update the frac frequencies counter. 
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

        #update the fracs with cached frequencies
        self.__frac_freqs.update(self.__freq_cache)

        return

    def finalize_counts(self):
        #call finalize counts on each MSM. Constructs a transition matrix from the counts added so far

        #check if counts are already final. Doing so again will mess with mass-weighting
        #so we do not allow this to be called more than once. 
        if self.__counts_finalized:
            warning = "Warning: Collection.finalize_counts() called after counts "
            warning+= "have been finalized. Nothing added since the first finalize "
            warning+= "will be reflected in the count or transition matrix."
            print(warning)
            return

        #finalize each MSM in the Collection
        for i in range(self.__num_elements):

            self.__MSM_map[i+1].finalize_counts(self.__macrostate_map)

        #set a flag that there are now transition matrices that can be used
        self.__counts_finalized = True

        return
    
    def set_MSM_map(self, map_index, msm):
        #place the given msm as the given index in the collection

        self.__MSM_map[map_index] = msm

        return

    def _set_finalized(self):
        #set flag that marks all the msms as finalized

        self.__counts_finalized = True
        return
    
    def make_reduced_collection(self, max_index_shift):
        '''
        Makes a reduced MSM such that the provided index in the discretization
        is the new monomer fraction 1. For example, consider the discretization
            D0 = Discretization([0,0.25,0.5,0.75,1])
        If we provide max_index_shift=1, then 0.75 becomes the new monfrac 1 and 
        everything else gets scaled for the new discretization
            D1 = Discretization([0, 0.33, 0.66, 1])
        The MSM used in [0,0.33] is the original MSM on [0,0.25].
        '''

        #if the shift is 0, just return the same collection
        if max_index_shift == 0:
            return self

        #first, we construct a new discretization
        new_max     = self.__discretization.get_cutoffs()[-1-max_index_shift]
        new_cutoffs = self.__discretization.get_cutoffs()[0:-max_index_shift]
        new_cutoffs = [bin/new_max for bin in new_cutoffs]
        new_disc    = Discretization(new_cutoffs)
        new_num_elements = new_disc.get_num_intervals()

        #now we make a new collection with this discretization
        new_C = Collection(new_disc, self.__macrostate_map, self.__parameters, 
                           self.__lag, self.__prune_tol, self.__verbose)
        
        #now we copy the relevant MSMs from self to the new Collection
        for i in range(new_num_elements):

            new_C.set_MSM_map(i+1, self.__MSM_map[i+1])
            
        #set the new collection to finalized, since it was built with finalized MSMs
        new_C._set_finalized()

        return new_C
    
    def performTPT(self, initial_indices, target_indices, max_size = None, 
                   cluster_size_threshold = 3, clusterize = True):
        #perform transition path theory calculations on each MSM in the collection

        #perform clusterization of each MSM unless otherwise specified
        if clusterize:
            self.clusterize_all(cluster_size_threshold)
                
        #compute the committor
        q = {}
        for i in range(self.__num_elements):
            q[i+1] = self.get_msm(i+1).compute_committor(initial_indices, target_indices)

        #compute critical nucleus
        c = {}
        for i in range(self.__num_elements):
            c[i+1] = self.get_msm(i+1).compute_critical_nucleus(initial_indices,    
                                        target_indices, self.__macrostate_map, max_size)
            
        return q, c
    
    def clusterize_all(self, cluster_size_threshold = 3):
        #perform clusterization on all sub-MSMs

        for i in range(self.__num_elements):
            self.__MSM_map[i+1].clusterize_matrix(self.__macrostate_map, 
                                                    cluster_size_threshold)
        
        self.__clusterized = True

        return

    
    def get_misses(self):

        return self.__map_misses
    
    def is_finalized(self):

        return self.__counts_finalized
    
    def is_clusterized(self):

        return self.__clusterized
    
    def get_frac_freqs(self):

        return self.__frac_freqs

    def get_lag(self):

        return self.__lag
    
    def get_num_states(self):

        return self.__num_states

    def get_discretization(self):

        return self.__discretization

    def get_map(self):

        return self.__macrostate_map

    def get_msm_index(self, monomer_fraction):

        return self.__discretization.determine_interval(monomer_fraction)

    def get_msm(self, msm_index):

        return self.__MSM_map[msm_index]

    def get_transition_matrix(self, msm_index, clusterized=False):

        return self.get_msm(msm_index).get_transition_matrix(clusterized)

    def get_count_matrix(self, msm_index):

        return self.get_msm(msm_index).get_count_matrix()
    
    def _print_all_matrix(self, mat_type, row, col):

        for i in range(self.__num_elements):
            
            #get the requested entries
            m = self.__MSM_map[i+1]
            if mat_type == "t":
                TM = m.get_transition_matrix()
            elif mat_type == "c":
                TM = m.get_count_matrix()
            
            #slice the proper row and col
            if TM.nnz > 0:
                if row is not None:
                    TM = TM.getrow(row)
                if col is not None:
                    TM = TM.getcol(col)

            #print out the values
            print("For m_frac in {}".format(self.__discretization.interval_index_to_bounds(i+1)),end="")
            print(" [Row = {}, Col = {}]".format(row, col))
            print(TM)

        return

    def print_all_transition_matrices(self, row=None, col=None):
        
        self._print_all_matrix("t", row, col)
        return

    def print_all_count_matrices(self, row=None, col=None):

        self._print_all_matrix("c", row, col)
        return
    
    def __fix_zero_one(self, mon_frac):
        #if the monomer fraction is exactly zero or 1, give it a slight push

        if mon_frac > (1-1e-6):
            mon_frac -= 1e-6

        elif mon_frac < 1e-6:
            mon_frac += 1e-6

        return mon_frac

    def get_effective_MSM(self):

        num_states = self.__num_states
        count_matrix = scipy.sparse.dok_matrix((num_states, num_states), dtype=int)

        for key in self.__MSM_map:

            count_matrix += self.get_msm(key).get_count_matrix()

        msm = MSM(num_states, lag=self.__lag)
        msm.set_count_matrix(count_matrix, self.__macrostate_map)

        return msm

    def compare_model(self, other_model, msm_index):
        #compare this models transition matrix to anothers

        #get the transition matrices to compare
        myTM = self.get_transition_matrix(msm_index)
        yoTM = other_model.get_transition_matrix(msm_index)

        #init storage for row norms across matrices
        differences = np.zeros(self.__num_states, dtype=float)

        #loop over each row and compute differences in distributions
        for state in range(self.__num_states):
            
            #get the row distribution in eahc matrix
            p = np.array(myTM[state,:].todense())[0]
            q = np.array(yoTM[state,:].todense())[0]

            #compute the summed abs error across the row
            diff = 0
            for dest_state in range(self.__num_states):
                if p[dest_state] > 0 and q[dest_state] > 0:
                    diff += np.abs(p[dest_state]-q[dest_state])
            differences[state] = diff

        #sort the differences greatest to least
        sorted_indices = np.argsort(differences)[::-1]

        #print out the distributions for each row
        for i in sorted_indices:
            print("Row {}, Diff {}".format(i, differences[i]))
            print(self.__macrostate_map.index_to_state(i))
            print()
            print(myTM[i,:])
            print()
            print(yoTM[i,:])
            print()

        return sorted_indices, differences

    def copy_model(self, other_model, msm_index, sorted_ids, num_to_copy):
        #provided with the output of compare model, copies the first 
        # 'num_to_copy' rows of the other model to current model

        myTM = self.get_transition_matrix(msm_index)
        yoTM = other_model.get_transition_matrix(msm_index)

        for sorted_id in sorted_ids[0:num_to_copy]:
            
            state = self.__macrostate_map.index_to_state(sorted_id)
            old = myTM[sorted_id,:]
            new = yoTM[sorted_id,:]

            replaced     = np.array(old.todense(),dtype=float)[0]
            replace_with = np.array(new.todense(),dtype=float)[0]
            self.__MSM_map[msm_index].set_row_prob(sorted_id, replace_with)
            print("Replaced row for index {}, {}".format(sorted_id, state))
            for i in range(len(replaced)):
                sz = self.__macrostate_map.index_to_state(i).get_size()
                print("Index {}, Size {}, old {}, new {}, ratio {}".format(i, sz, replaced[i], replace_with[i], replace_with[i]/replaced[i]))

        return
    
    def prune_row(self, msm_index, row, prune_tol):
        #prune the given row of the given MSM of all probs less than tol

        myTM = self.get_transition_matrix(msm_index)
        current_row = np.array(myTM[row,:].todense(),dtype=float)[0]

        #do pruning
        current_row[current_row < prune_tol] = 0

        #renormalize
        row_sum = np.linalg.norm(current_row, 1)
        current_row /= row_sum
        
        #replace row
        self.__MSM_map[msm_index].set_row_prob(row, current_row)

        return






    
class MultiMSMSolver:

    def __init__(self, multiMSM):
        
        #keep a copy of the multiMSM to reference during methods
        self.__multiMSM = multiMSM

        #check that the MultiMSM has been finalized so solutions can be computed
        self.__check_finalized()

        #extract some commonly used constant objects from the multiMSM
        self.__num_states = multiMSM.get_num_states()
        self.__discretization = multiMSM.get_discretization()
        self.__macrostate_map = multiMSM.get_map()

        #store info on monomers
        self.__monomer_state = Monomer().get_state()
        self.__monomer_index = Monomer().get_index()

        #init storage for solutions to forward and backward equations
        #size of these arrays depend on final time, a runtime variable
        self.__current_p0 = -1
        self.__fke_soln = None
        self.__bke_soln = None

        self.__entropy_production = []
        self.__entropy_time = []

        #the backward equation needs to know what transition matrices were used
        #to solve the forward equation. This dict will store the indices and weights
        #to reconstruct the matrix as tuples for each lag, which indexes the dict
        self.__TM_indices = dict()

        return


    def solve_FKE(self, T, p0 = None, frac = 0.25, compute_entropy = False):
        '''
        Solve FKE up to integer lag T. p0 is the initial distribution, which will
        default to 100% monomers if another distribution is not specified. 

        The optional parameter frac will perform smoothing of the solution 
        across monomer fraction bin edges if a positive value between 0 and 1 
        is specified. If the monomer fraction is within L*frac of bin edge, where 
        L is the bin width, it construct a linear combination of transition matrices
        for the current and neighboring bin. 

        We have found that 0.25 provides good results for each system we have tested
        on. 
        '''

        #verify and setup the initial conditions
        init_dist = self.__setup_initial_condition(p0)

        #setup soln to FKE. if re-solving with same p0, return or resize 
        #depending on T
        start_time = self.__setup_FKE(init_dist, p0, T)
        current_mon_frac = self.__fke_soln[start_time, self.__monomer_index]

        #init needed variables for smoothing if requested
        if frac > 0:
            self.__transition_regions = self.__make_frac_bins(frac, self.__discretization.get_cutoffs())
        
        #solve the FKE, grabbing the relevant transition matrix each iteration
        for t in range(start_time, T):

            #get a transition matrix for this mon frac using the appropriate method
            if frac > 0:
                TM = self.__get_matrix_LC_frac(current_mon_frac, t)
            else:
                TM = self.__get_matrix_base(current_mon_frac, t)

            #check if entropy production is requested
            if compute_entropy and t % int(compute_entropy) == 0:
                e = self.__compute_entropy_prod(TM, self.__fke_soln[t,:])
                self.__entropy_production.append(e)
                self.__entropy_time.append(t)

            #update the probabilities 1 step in future
            self.__fke_soln[t+1, :] = self.__fke_soln[t, :] * TM

            #get the index for the next transition matrix from monomer frac
            current_mon_frac = self.__fke_soln[t+1,self.__monomer_index]

        #store what p0 was used to compute this soln
        self.__current_p0 = p0

        #return a copy of the full solution
        return self.__fke_soln.copy()

    def get_FKE(self, T, p0 = None, frac = 0.25, target_index_list = None, 
                compute_entropy = False):
        #return the solution to the FKE up to the specified time and given 
        #initial distribution 
        #if target_index set is None, return full thing. Otherwise return the
        #summed probability of the target set

        #check that p0 is same as stored, otherwise compute full soln
        if self.__current_p0 != p0 or (self.__fke_soln is not None and T+1 > self.__fke_soln.shape[0]):
            self.solve_FKE(T, p0=p0, frac=frac, compute_entropy=compute_entropy)

        #check if target index is not None and sum the probabilities
        if target_index_list is not None:
            if not isinstance(target_index_list, list):
                target_index_list = [target_index_list]

            return self.__fke_soln[:,target_index_list].sum(1)

        #return the full solution
        return self.__fke_soln
    
    def get_entropy(self):
        #return the entropy production as function of time

        if len(self.__entropy_production) == 0:
            err = "Entropy Production has not been computed. Call solve_FKE() with "
            err+= "compute_entropy=True parameter before calling get_entropy(). "
            raise RuntimeError(err)
        
        return self.__entropy_time, self.__entropy_production

    def solve_BKE(self, T, target_index_list):
        '''
        Solve the backward kolmogorov equation beginning at time T+1 and going back
        to time 0. Return full time dependent solution matrix.

        It is assumed that the reward function for this equation is the indicator
        vector for a target set of states, as its intended use is to maximize the
        associated probability. The solution at T+1 will be init with 1s for states
        in target_index_list, and 0s otherwise. 

        TODO: generalize to other reward functions?

        Note: The backwards propogation requires that the forward equation has been 
        solved to know which transition matrix to use. 

        '''

        #init, error check, and set up the final condition for the BKE
        self.__setup_BKE(T, target_index_list)

        #Solve BKE backwards in time
        for t in range(T-1, -1, -1):

            #get the transition matrix from forward evolution
            TM = self.__reconstructTM(t)

            #update the soln to previous time
            self.__bke_soln[:, t] = TM * self.__bke_soln[:, t+1] 


        return self.__bke_soln.copy()

    def verify_invariance(self, verbose=False):
        #verify that the inner product of forward and backward solutions are
        #invariant in time

        #check that forward and backward solutions exist
        if self.__fke_soln is None or self.__bke_soln is None:
            err_msg = "Forward and Backward equations have not been solved"
            raise RuntimeError(err_msg)
        
        #init storage for the time dependent inner product
        T = min(self.__fke_soln.shape[0], self.__bke_soln.shape[1])
        invariant_product = np.zeros(T, dtype=float)
        S = 0 # sum consecutive differences

        for t in range(T):

            f_vec = self.__fke_soln[t,:]
            b_vec = self.__bke_soln[:,t]
            invariant_product[t] = np.inner(f_vec, b_vec)
            
            if t>0:
                S += np.abs(invariant_product[t]-invariant_product[t-1])

        invariant = (S < 1e-10)
        print("Testing Invariance of Inner Product...")
        print("Total deviation: {}, Invariant: {}".format(S, invariant))
        if verbose:
            print("Inner product:\n",invariant_product)
     
        return
    
    def generateTrajectory(self, T, start_index = 0, p0 = None, output_type = "index",
                           clusterized=True):
        '''
        Generates a trajectory of length T according to the MultiMSM dynamics. 
        Begins in starting state and draws a random new state from the row
        of the corresponding transition matrix at each lag time. 

        Relies on the sequence of transition matrices stored during a forward solve. 
        If a forward solve has not been performed until time T, will perform one in 
        the initialization. 

        Output trajectory is a list of sequential states, reported as either indices,
        State objects, or cluster sizes. 
        '''

        #check if transitions matrices have been clusterized if requested
        if clusterized and not self.__multiMSM.is_clusterized():
            self.__multiMSM.clusterize_all()

        #check if a forward solve has been performed to the desired time. do it if not
        self.solve_FKE(T, p0=p0)

        #init storage for trajectory
        traj = [start_index]
        current_index = start_index

        #loop over all times to generate new states
        for i in range(T):

            #grab the transition matrix at this time
            TM = self.__reconstructTM(i, clusterized)

            #extract the row corresponding to the current state
            row = TM[current_index, :]
            row.sort_indices()
            row.eliminate_zeros()

            # Get the indices of nonzero values and their values, convert to arrays
            nonzero_indices = np.array(row.nonzero()).squeeze()[1]
            nonzero_values  = np.array(row.data).squeeze()
            
            #sample this distribution to generate new state
            if not isinstance(nonzero_indices, np.ndarray):
                current_index = nonzero_indices
            else:
                current_index = np.random.choice(nonzero_indices, p=nonzero_values)
            traj.append(current_index)

        #post process the trajectory to the correct output format
        if output_type == "size":
            for i in range(len(traj)):
                traj[i] = self.__macrostate_map.index_to_state(traj[i]).get_size()
        return traj
    
    def __compute_entropy_prod(self, TM, distribution):
        '''
        Determine the entropy production from the current transition matrix and    
        probability distribution. Formula is
            sum_{i,j} (p_i T_ij - p_j T_ji) log(p_iT_ij/p_jT_ji)
        or in words, the difference in forward and backward flux times the log of the 
        ratio of forward and backward flux. 
        '''

        #get all entries that are nonzero at this time
        non_zero_ids = np.where(distribution > 1e-6)[0]

        #init the entropy sum
        ep = 0
        pos_tol = 1e-8

        #loop over all pairs of states
        print(len(non_zero_ids))
        for idx in range(len(non_zero_ids)):
            for jdx in range(idx+1, len(non_zero_ids)):

                i = non_zero_ids[idx]
                j = non_zero_ids[jdx]
                
                forward = distribution[i] * TM[i,j]
                backward= distribution[j] * TM[j,i]

                if forward > pos_tol and backward > pos_tol:
                    ep += (forward-backward) * np.log(forward/backward)

        return ep
    

    def __setup_initial_condition(self, p0):
        #determine the IC and do type and bounds checking

        #set the IC from user input
        if p0 is None:
            init_dist = self.__FKE_monomer_start()
            print("Warning: Initial distribution not specified. Defaulting to 100% monomer")
        elif p0 == "monomer_start" or p0 == "ms":
            init_dist = self.__FKE_monomer_start()
        else:
            init_dist = p0

        #first check the distribution is an nparray and make it one if not
        if type(init_dist) is not np.ndarray:
            init_dist = np.array(init_dist, dtype=float)

        #check that it is over the correct number of states
        if len(init_dist) != self.__num_states:
            err_msg =  "The length of the supplied initial distribution ({})".format(len(init_dist))
            err_msg += " does not match the number of states ({})".format(self.__num_states)
            raise ValueError(err_msg)

        return init_dist

    def __FKE_monomer_start(self):
        #return an initial distribution corresponding to 100% subunits as monomers

        init_dist = np.zeros(self.__num_states, dtype=float)
        init_dist[self.__monomer_index] = 1.0

        return init_dist

    def __setup_FKE(self, init_dist, p0, T):
        #setup data structure to hold FKE soln
        #if it already exists and used the same p0 then either...
        #   1) return if T < size solution
        #   2) resume from the end of T > size solution

        #the return is the time lag to start the solve at

        #if the p0 is different setup a fresh solve
        if self.__current_p0 != p0:
            self.__fke_soln = np.zeros((T+1, self.__num_states), dtype=float)
            self.__fke_soln[0, :] = init_dist

            #clear the index dict
            self.__TM_indices.clear()
            return 0

        #if we get here, p0 is the same. Check the T values

        #if we have already solved longer than the requested time
        if self.__fke_soln.shape[0] >= T+1:
            return T #Note: used to be T+2, but I think this was wrong

        #if we get here, we need to continue the solve, return len of soln
        return self.__fke_soln.shape[0]

    def __setup_BKE(self, T, target_index_list):
        #check that conditions are satisfied to do backwards solve and set up

        #check that the forward equation is solved
        if self.__fke_soln is None or self.__current_p0 == -1:
            raise RuntimeError("FKE solution is missing. Solve the FKE first.")

        #check that the forward equation is solved to the requested final time
        solved_to = self.__fke_soln.shape[0]
        if solved_to <= T:
            err_msg = "FKE solved to lag {}. ".format(solved_to-1)
            err_msg+= "BKE requested at lag {}. ".format(T)
            err_msg+= "FKE needs to be solved longer. "
            raise RuntimeError(err_msg)

        #ensure target indices are in a list for vectorized init
        if not isinstance(target_index_list, list):
            target_index_list = [target_index_list]

        #check that the list is non-empty, give warning of trivial soln
        if len(target_index_list) == 0:
            err_msg = "No target states have been supplied. This will result "
            err_msg+= "in a trivial zero solution, which we assume is unintended."
            raise RuntimeError(err_msg)

        #initialize the BKE solution w/ 1s in target indices
        self.__bke_soln = np.zeros((self.__num_states, T+1), dtype='float')
        self.__bke_soln[target_index_list, T] = 1.0

        return


    def __make_frac_bins(self, frac, cutoffs):
        #for each divider, create an interval that is [d-frac*W_l,d+frac*W_r] where
        #d is the divider position, and W_i is the width of the bin to the left or right
        #of the divider. return a list of these intervals, including the div point


        transition_regions = []

        for i in range(1,len(cutoffs)-1):

            left_div  = cutoffs[i-1]
            div       = cutoffs[i]
            right_div = cutoffs[i+1]

            l_w = div - left_div
            r_w = right_div - div

            region = [div - frac*l_w, div, div+frac*r_w]
            transition_regions.append(region)

        return transition_regions

    def __get_matrix_base(self, current_mon_frac, t):
        #simply grab the transition matrix from this interval

        index = self.__multiMSM.get_msm_index(self.__fix_zero_one(current_mon_frac))
        TM = self.__multiMSM.get_transition_matrix(index)

        #add the index of this matrix to the indices dict
        self.__TM_indices[t] = [(index,1)]

        return TM
    
    def __get_matrix_LC_frac(self, current_mon_frac, t):
        #get a transition matrix for the current monomer fraction
        #this method checks if mon frac is within given frac of the current bin
        #width of the divider and contructs LC of neighboring matrices with that weight

        #check if current mon frac is in any of the transition regions
        in_region = False
        region_id = -1
        for i in range(len(self.__transition_regions)):
            region = self.__transition_regions[i]

            if current_mon_frac > region[0] and current_mon_frac < region[2]:
                in_region = True
                region_id = i
                # print(current_mon_frac, region)
                break

        #if not in any region, return base TM
        if not in_region:
            return self.__get_matrix_base(current_mon_frac, t)

        #if we are in a region, construct the LC of transition matrices
        a = region[0]
        b = region[2]
        c = region[1]

        #determine alpha based on which side of the divider we are and its size
        if current_mon_frac < c and current_mon_frac > a:
            alpha = 0.5 * (current_mon_frac-a) / (c-a)
        else:
            alpha = 0.5 + 0.5 * (current_mon_frac-c) / (b-c)

        #get the transition matrices to the left and right and compute LC
        leftTM = self.__multiMSM.get_transition_matrix(region_id+1)
        rightTM = self.__multiMSM.get_transition_matrix(region_id+2)
        TM = alpha * rightTM + (1-alpha) * leftTM

        #add the linear combination for this matrix to the index dict
        self.__TM_indices[t] = [(region_id+1,1-alpha),(region_id+2,alpha)]
        return TM
    
    def __fix_zero_one(self, mon_frac):
        #if the monomer fraction is exactly zero or 1, give it a slight push

        if mon_frac > (1-1e-6):
            mon_frac -= 1e-6

        elif mon_frac < 1e-6:
            mon_frac += 1e-6

        return mon_frac

    def __check_finalized(self):
        #check if the MultiMSM has been finalized before computing solns

        if not self.__multiMSM.is_finalized():

            err_msg =  "Transition matrices have not been finalized. Solution to FKE "
            err_msg += "cannot be computed yet"
            raise RuntimeError(err_msg)

        return

    def __reconstructTM(self, t, clusterized = False):
        #reconstruct transition matrix that was used to solve the forward eqn
        #from t to t+1

        #get the list of weights and indices
        factors = self.__TM_indices[t]
        c = clusterized

        #if there is only one matrix, return that
        if len(factors) == 1:
            return self.__multiMSM.get_transition_matrix(factors[0][0], c)

        #otherwise, construct the linear combination
        left = self.__multiMSM.get_transition_matrix(factors[0][0], c) * factors[0][1]
        right= self.__multiMSM.get_transition_matrix(factors[1][0], c) * factors[1][1]

        return left + right
    

        

        
        

