import sys
import os
import fnmatch
import inspect


from collections import defaultdict


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

    def get_state(self):

        return self.__state

    def get_index(self):

        return self.__index







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

    def __init__(self, monomer_state = None, load_name = None):

        #check for disallowed inputs
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

        #check if the directories /data/ and /data/sampling/ exist
        if not os.path.isdir("data/"):
            os.makedirs("data/")

        if not os.path.isdir("data/map/"):
            os.makedirs("data/map/")

        #grab the parameter identifiers from the data folder and set storage loc
        self.__storage_location  = "data/map/"

        return

    def __handle_input(self, monomer_state, load_name):
        #check for disallowed case of neither parameter being provided

        if monomer_state is None and load_name is None:
            err_msg  = "The MacrostateMap needs to be initialized with either a description "
            err_msg += "of the monomer state, or the name of a saved mapping. "
            raise RuntimeError(err_msg)

        return

    def __handle_load(self, load_name):
        #check if loading is requested. do load if so, and check for errors

        if load_name is not None:
            try:

                self.__load(load_name)
                return True


            except:

                filepath = self.__storage_location + load_name + ".pkl"
                err_msg  = "Object could not be loaded from {}. ".format(filepath)
                err_msg += "Check the path and try again, or contruct a new map from scratch"
                raise RuntimeError(err_msg)

        return False

    def __handle_setup(self, monomer_state):
        #perform first time setup of a new map

        #init dicts for the mappings
        self.__toIndex = defaultdict(lambda: None)
        self.__toState = defaultdict(lambda: None)

        #init and set monomer state and index, creates a singleton for Monomer
        self.__monomer_state = None
        self.__monomer_index = -1
        self.set_monomer_state(monomer_state)

        #set load state to false
        self.__was_loaded = False

        return


    def save(self, map_name):

        #set the path to save the object
        self.__map_name = map_name
        filepath        = self.__storage_location + map_name + ".pkl"

        #pickle it to the location
        with open(filepath,'wb') as outfile:
            pickle.dump(self, outfile)

        return

    def __load(self, map_name):

        #set the filepath and load it
        filepath = self.__storage_location + map_name + ".pkl"

        with open(filepath,'rb') as infile:
            self.__dict__ = pickle.load(infile).__dict__
            self.__been_updated = False
            self.__was_loaded   = True

            # TODO - may need to create Monomer singelton?

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

    def update_maps(self, state):
        #check if the supplied state is in the map. If not add it

        if state not in self.__toIndex:

            L = self.get_num_states()

            self.__toIndex[state] = L
            self.__toState[L] = state

            print("State {} added to map with index {}".format(state, L))

            self.__been_updated = True
        
        return

    def state_to_index(self, state):

        return self.__toIndex[state]

    def index_to_state(self, index):

        return self.__toState[index]

    def get_num_states(self):

        return len(self.__toIndex)

    def get_monomer_state(self):

        return self.__monomer_state

    def get_monomer_index(self):

        return self.__monomer_index