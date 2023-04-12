from SAASH.util.state import StateRefCollection
from SAASH.util.state import StateRef
from SAASH.util.state import State

from ..MSM.MSM import MSM
from ..MSM.MultiMSM import Collection

from ..util.state import MacrostateMap

import os, sys
import pickle

import numpy as np

from collections import defaultdict
import matplotlib.pyplot as plt

#CRITICAL: If things are pickled within the util module of SAASH, 
#the following is required to unpickle outside of that project's namespace
# from SAASH import util
# sys.modules['util'] = util

#######################################################
############### Observations per state ################
#######################################################

class StateCountHist:

    def __init__(self, msm, macrostate_map):

        #ensure the input passed is an MSM object
        if not isinstance(msm, MSM):
            raise TypeError("Please provide an MSM instance as input 0")

        #ensure the second input us a macrostate map
        if not isinstance(macrostate_map, MacrostateMap):
            raise TypeError("Please provide a MacrostateMap instance to input 1")
        self.__map        = macrostate_map
        self.__num_states = macrostate_map.get_num_states()
        
        #define dicts to store the distributions from data
        self.__counts = msm.get_row_counts()
        self.__count_hist = defaultdict(int)

        #process the MSM to get count data
        self.__process_MSM(msm)


    def __process_MSM(self, msm):
        #invert the counts per state data to get states per count

        for i in range(self.__num_states):

            self.__count_hist[self.__counts[i]] += 1


    def plot_hist(self, thresholds = [0,100,1000]):
        #make a histogram showing how many states are in each sampling bin

        #seperate the counts by the threshold bins
        new_hist = self.__separate_by_threshold(thresholds)

        #define the count ranges for each bar
        labels = []
        for i in range(len(thresholds)-1):
            labels.append(str(thresholds[i])+'-'+str(thresholds[i+1]))
        labels.append(str(thresholds[-1])+"+")
        
        #make a bar plot of the data and set radable font sizes
        plt.bar(range(len(thresholds)), new_hist, tick_label=labels)
        plt.xlabel("# observations",fontsize=18)
        plt.xticks(fontsize=12)
        plt.ylabel("# states",fontsize=18)
        plt.yticks(fontsize=12)
        plt.show()

        return labels, new_hist

    def __separate_by_threshold(self, thresholds):

        new_hist = np.zeros(len(thresholds), dtype=int)
        for count,states in self.__count_hist.items():

            #determine which bin count falls in 
            bin_num = np.digitize(count, thresholds)-1

            #augment this index in the array bu the number of states with this count
            new_hist[bin_num] += states

        return new_hist



class MonomerFractionCountHist:

    def __init__(self, traj_folder):

        self.__count_hist     = defaultdict(int)
        self.__monomer_counts = defaultdict(int)
        self.__cache_files    = []

        self.__find_cache_files(traj_folder)
        self.__process_cache_files()



    def __find_cache_files(self, traj_folder):
        #gather all of the analysis cache files for trajs in the given folder

        for subdir, dirs, files in os.walk(traj_folder):
            for file in files:

                if file.endswith('.cache'):
                    self.__cache_files.append(os.path.join(subdir, file))

        #check that there is at least 1 file before continuing 
        if len(self.__cache_files) == 0:
            err_msg = "No cache files were found in subdirectories of {}\n"
            err_msg+= "Have you constructed a MultiMSM with caching enabled?"
            raise RuntimeError(err_msg)

        return

    def __process_cache_files(self):
        #loop over all cache files and add counts to them

        for cache_file in self.__cache_files:
            with open(cache_file,'rb') as infile:

                save_data = pickle.load(infile)
                count_cache = save_data[0]

                self.__add_counts_from_cache(count_cache)

        return


    def __add_counts_from_cache(self, count_cache):
        #loop over all counts, add to hist dicts

        for transition in count_cache:

            #extract the indices and the inner dict with frac data
            state1 = transition[0]
            state2 = transition[1]
            data = count_cache[transition]

            #set flag if the transition is monomer to monomer 
            is_monomer_monomer = False
            if state1 == 0 and state2 == 0:
                is_monomer_monomer = True

            #loop over all fracs that show this transition and add counts
            for frac in data:

                #get counts, compute an index from frac, add the counts
                counts = data[frac]
                
                if is_monomer_monomer:
                    self.__monomer_counts[frac] += counts
                else:
                    self.__count_hist[frac] += counts

        return
                
    def plot_hist(self, hide_large = False):
        #plot histograms for transition counts as function of monomer frac
        #plots monomer to monomer and others separately

        #TODO: apply a filtering if hide large was set

        #TODO: combine bins so there are not 100 of them in the hist

        #make a subplot figure
        fig, (ax1,ax2) = plt.subplots(1,2)


        ax1.bar(self.__count_hist.keys(), self.__count_hist.values())
        ax2.bar(self.__monomer_counts.keys(), self.__monomer_counts.values())

        #TODO: format the plots so they look nice
        plt.show()

        return




