import sys
import os
import fnmatch
import inspect
import pickle

from collections import defaultdict

'''
This will require installation of the SAASH library to use this software.
In the future, will need to either copy-paste the finalized version of State,
or publish SAASH on pip in order to set it as a requirement.
'''

from SAASH.util.state import State


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):

        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)

        return cls._instances[cls]


class Monomer(metaclass=Singleton):

    '''
    Singleton to represent the monomer of the system. Used for accessing the index
    corresponding to the monomer whererver it is needed. 

    state is a description of the monomer state, i.e. "1" or a single node graph
    index is the mapped macrostate index of the monomer, which should be 0, but it tracked for 
    completeness and clarity. 
    '''
    
    def __init__(self, state, index):

        self.__state = state
        self.__index = index

        if not isinstance(state, State):
            err_msg = "Init the Monomer with a State object"
            raise RuntimeError(err_msg)

    def get_state(self):

        return self.__state

    def get_index(self):

        return self.__index

    def get_size(self):

        return 1



class MacrostateMap:

    '''
    Mapping from a state description, i.e. 3 for trimer, the canonical graph representation, 
    or a tuple of coordinates (3,2), into a unique index to describe the state. The indices are assigned
    in the order the states are seen in trajectory data analysis. 

    Uses dicts to convert between states and indices. Unknown states can be added to the map as they are found

    Must be init with a description of the monomer state. Creates the first Monomer singleton instance. 

    Can be saved to disk and loaded for future use, with options for applying updates. 

    Assumed to be stored in the /data/map/ directory. Name can be given to the map. 
    '''

    # TODO : test how singletons behave when loading. may need to construct a new one on load. 

    def __init__(self, monomer_state = None, load_name = None, verbose=False):

        #check for disallowed inputs
        self.__verbose = verbose
        self.__handle_input(monomer_state, load_name)

        #set the default save location
        self.__set_save_loc()

        #check if loading is requested
        loaded = self.__handle_load(load_name)

        #init the system manually if not loading
        if not loaded:
            self.__handle_setup(monomer_state)

        return

    def __del__(self):
        #if deleting, and the map has been updated without being saved, save it. 

        if self.__was_loaded and self.__been_updated:
            self.save(self.__map_name)

        return

    def __set_save_loc(self):
        #check if the default directory exists and set reference to it

        #check if the directories /data/ and /data/map/ exist
        if not os.path.isdir("data/"):
            os.makedirs("data/")

        if not os.path.isdir("data/map/"):
            os.makedirs("data/map/")

        #grab the parameter identifiers from the data folder and set storage loc
        self.__storage_location  = "data/map/"

        return

    def __handle_input(self, monomer_state, load_name):
        #check for case where no parameters are specified

        #set flag for using default monomer
        self.__use_default_monomer = False

        #provide a warning that the default monomer is being used if verbose
        if monomer_state is None and load_name is None:
            self.__use_default_monomer = True
            if self.__verbose:
                warning_msg = "WARNING: no monomer state provided. Using the default. "
                print(warning_msg)


        return

    def __handle_load(self, load_name):
        #check if loading is requested. do load if so, and check for errors

        if load_name is not None:
            try:

                self.__load(load_name)
                return True


            except:

                filepath = self.__storage_location + load_name + ".map"
                err_msg  = "Object could not be loaded from {}. ".format(filepath)
                err_msg += "Check the path and try again, or contruct a new map from scratch"
                raise RuntimeError(err_msg)

        return False

    def __handle_setup(self, monomer_state):
        #perform first time setup of a new map

        #TODO:check for list of monomer states 

        #init dicts for the mappings
        self.__toIndex = defaultdict(lambda: None)
        self.__toState = defaultdict(lambda: None)

        #init and set monomer state and index, creates a singleton for Monomer
        self.__monomer_state = None
        self.__monomer_index = -1

        #if using default monomers
        if self.__use_default_monomer:
            dM = Monomer(State(1,{}),0)
            self.set_monomer_state(dM.get_state())

        #if using provided monomers
        else:
            self.set_monomer_state(monomer_state.get_state())

        #set load state to false
        self.__was_loaded = False

        return


    def save(self, map_name):

        #set the path to save the object
        self.__map_name = map_name
        filepath        = self.__storage_location + map_name + ".map"

        #pickle it to the location
        with open(filepath,'wb') as outfile:
            pickle.dump(self, outfile)

        return

    def __load(self, map_name):

        #set the filepath and load it
        filepath = self.__storage_location + map_name + ".map"

        with open(filepath,'rb') as infile:
            self.__dict__ = pickle.load(infile).__dict__
            self.__been_updated = False
            self.__was_loaded   = True

            #create the monomer singleton now so it exists. no issue if redundant
            Monomer(self.__monomer_state, self.__monomer_index)

        return


    def set_monomer_state(self, monomer_state):
        '''
        Set what the "monomer" state is considered to be. This could be the value "1" if 
        tracking number of subunits, or the representation of a graph with 1 node if tracking
        graph structures. 

        This sets the state, gives it an index (which should be 0), and creates a singleton
        object called Monomer, which allows access to the state and index info globally
        '''

        self.__monomer_state = monomer_state
        if monomer_state is not None:
            self.update_maps(monomer_state)
            self.__monomer_index = self.state_to_index(self.__monomer_state)
        else:
            raise ValueError("Please provide a description of the monomer state")

        #create Monomer singleton
        Monomer(self.__monomer_state, self.__monomer_index)

        return

    def update_maps(self, state, verbose = False):
        #check if the supplied state is in the map. If not add it

        if state not in self.__toIndex:

            L = self.get_num_states()

            self.__toIndex[state] = L
            self.__toState[L] = state

            if verbose:
                print("{} added to map with index {}".format(state, L))

            self.__been_updated = True
        
        return

    def state_to_index(self, state):

        if state in self.__toIndex:
            return self.__toIndex[state]
        
        return -1

    def index_to_state(self, index):

        if index < self.get_num_states():
            return self.__toState[index]
        
        return None

    def get_num_states(self):

        return len(self.__toIndex)

    def get_monomer_state(self):

        return self.__monomer_state

    def get_monomer_index(self):

        return self.__monomer_index

    def filter_by_size(self, sizes, verbose = False):
        #return all indices for states with size in sizes

        #if only an int is given, put in list so code is general
        if isinstance(sizes, int):
            sizes = [sizes]

        #init output list for indices
        output_indices = []

        #perform dictionary comprehension for each size in sizes
        for size in sizes:

            indices = [self.__toIndex[state] for state in self.__toIndex.keys() if state.get_size() == size ]
            output_indices += indices
            
            if verbose:
                print("Listing all states of size {}...".format(size))
                for index in indices:
                    print("Index {}, {}".format(index, self.__toState[index]))
                print()

        return output_indices
    

class TargetStates:

    '''
    This class keeps track of the state or states of interest for a particular 
    calculation. Has arrays of the state indices and can construct the State from 
    this.

    States can be manually added, either by State or by index. There is also a query 
    option that takes in a state size, outputs all states indices with that size, and 
    then asks the user if they want to add any states to the class. 
    '''

    def __init__(self, MM, target_indices = []):
        
        #store the map for later state indexing
        self.__macrostate_map = MM

        #create set to store indices of states added
        self.__target_indices = set()
        self.add_indices(target_indices)

        return
    
    def add_indices(self, indices):
        #adds supplied indices to the target set

        for index in indices:
            self.__target_indices.add(index)

        return
    
    def get_indices(self):

        return list(self.__target_indices)
    
    def get_states(self):

        return [self.__macrostate_map.index_to_state(index) for index in self.__target_indices]
    
    def query(self, size):
        #query the macrostate map for all states of a given size. Prompt user to add
        #their desired states from this list

        #print out all states of this size
        self.__macrostate_map.filter_by_size(size, verbose=True)

        #ask the user for list of indices to add
        msg = "\nProvide a space separated list of indices you would like to add...\n"
        to_add = input(msg)

        #parse the user input at spaces
        to_add = to_add.split(" ")

        #try to convert every entry to int, give error if fail
        try:
            to_add = [int(index) for index in to_add]
        except:
            err_msg = "Could not convert user input to int. Make sure you supply a "
            err_msg+= "space separated list of integers."
            raise TypeError(err_msg)
        
        #on success, add to the indices
        self.add_indices(to_add)

        return
        






