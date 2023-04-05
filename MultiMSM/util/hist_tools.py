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




