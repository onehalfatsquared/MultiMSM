'''
Defines a class to manage analysis of brute-force sampling data, i.e. to compare
MSM estimates to sampled averages for things like probabilities of each cluster 
size. 

Implements a save and load system so the analysis does not have to be repeated each 
time one wants to visualize the sampled data. To do so, Data classes will be pickled 
into the directory data/sampling/, which will be a subdirectory of the directory this
file is located. 

If the class is loaded, it compares all previously analyzed files to the files
in the data folder. It will process any new ones and add them. Can also recompute,
which deletes all old data and does the analysis from scratch. 

It is assumed that all trajectory files will be stored in the folder 
"../trajectories/parameters/", where "parameters" gives an identifier of what tunable parameters
the trajectories were run for. E.g. "../trajectories/E10.0/".  

Cluster size trajectory files are assumed to have file extension ".sizes"

'''

import sys
import os
import fnmatch
import glob

import numpy as np

import pickle

from enum import Enum

class SizeCompare(Enum):

    SAME = 0
    TIME = 1
    SIZE = 2


class SizeDistribution:

    def __init__(self, cluster_size_counts = None, largest_size = 60):

        self.__largest_size = largest_size
        self.__num_time_pts = 1

        self.__TIME_DIM = 0
        self.__SIZE_DIM = 1

        if cluster_size_counts is not None:
            self.__setup_arrays(cluster_size_counts)
        else:
            self.__size_counts = np.zeros((self.__num_time_pts, self.__largest_size), dtype=float)


    def __setup_arrays(self, cluster_size_counts):
        #grab array information from the size counts

        #check that the supplied array is an nparray with suitable dimensions
        if not isinstance(cluster_size_counts, np.ndarray):
            raise TypeError("Please provide an np.ndarray for cluster_size_counts")

        #set the size propoerties
        self.__num_time_pts = cluster_size_counts.shape[0]
        self.__largest_size = cluster_size_counts.shape[1]

        #store the array
        self.__size_counts  = cluster_size_counts.copy()

        return

    def get_num_time_points(self):

        return self.__num_time_pts

    def get_largest_size(self):

        return self.__largest_size

    def get_cluster_size_data(self):

        return self.__size_counts.copy()



    def __add__(self, other_distribution):

        #check for consistency between the sizes and fix discrepancies
        change_list = self.__get_resize_cases(other_distribution)
        for change in change_list:
            self.__apply_fix(other_distribution, change)

        #now that the arrays are the same size, add them
        self.__size_counts += other_distribution.get_cluster_size_data()

        #update the array size stats
        self.__num_time_pts = self.__size_counts.shape[0]
        self.__largest_size = self.__size_counts.shape[1]

        return self


    def __get_resize_cases(self, other_distribution):
        #return a list consisting of all of the changes necessary to make arrays same size
        #tuples consisting of (type of change (enum), which array needs to be changed (str))

        change_list = []

        if (self.__size_counts.shape == other_distribution.get_cluster_size_data().shape):
            change_list.append((SizeCompare.SAME, None))

        if (self.get_largest_size() > other_distribution.get_largest_size()):
            change_list.append((SizeCompare.SIZE,"other"))

        elif (self.get_largest_size() < other_distribution.get_largest_size()):
            change_list.append((SizeCompare.SIZE,"self"))

        if (self.get_num_time_points() > other_distribution.get_num_time_points()):
            change_list.append((SizeCompare.TIME,"other"))

        elif (self.get_num_time_points() < other_distribution.get_num_time_points()):
            change_list.append((SizeCompare.TIME,"self"))

        return change_list

    def __apply_fix(self, other_distribution, change):
        #apply the supplied change to the distributions

        #determine which case we are handling
        case      = change[0]
        to_modify = change[1]

        #exit if case is same
        if (case == SizeCompare.SAME):
            return

        #handle size cases
        if (case == SizeCompare.SIZE):

            if to_modify == "self":
                new_size = other_distribution.get_largest_size()
                self.resize_array(new_size, self.__SIZE_DIM)

            elif to_modify == "other":
                new_size = self.get_largest_size()
                other_distribution.resize_array(new_size, self.__SIZE_DIM)

        #handle time cases
        if (case == SizeCompare.TIME):

            if to_modify == "self":
                new_size = other_distribution.get_num_time_points()
                self.resize_array(new_size, self.__TIME_DIM)

            elif to_modify == "other":
                new_size = self.get_num_time_points()
                other_distribution.resize_array(new_size, self.__TIME_DIM)

        return


    def resize_array(self, new_size, dim):
        #resize the array to the new size in the specified dimension and fill with 0s
        #NOTE: prior logic should ensure that arrays are only ever made larger

        #get the current shape and size of the specified dim
        array_shape = self.__size_counts.shape
        old_size    = array_shape[dim] 

        #pad the array with zeros
        pad_width = np.zeros((2,2), dtype=int)
        pad_width[dim][1] = new_size-old_size
        self.__size_counts = np.pad(self.__size_counts, pad_width, 
                                    'constant', constant_values=(0,0))
        return


class ClusterSizeData:

    def __init__(self, data_folder, recompute = False):

        #set location for storage of this class based on name of data folder
        self.__set_storage_info(data_folder)

        #unless recompute, try to load existing class object
        if not recompute:
            try:

                self.__load(self.__storage_location)

            except:

                print("Could not load object at {}. Computing from scratch.".format(self.__storage_location))
                self.__init_log()

        else:

            self.__init_log()

        #process the files in the data folder
        self.__process_files(data_folder)

        return


    def __del__(self):
        #if deleting, and the map has been updated without being saved, save it. 

        if self.__was_loaded and self.__been_updated:
            self.__save()

        return


    def __save(self):

        with open(self.__storage_location,'wb') as outfile:
            pickle.dump(self, outfile)
            print("Cluster Size Data saved to {}".format(self.__storage_location))

        return

    def __load(self, filepath):

        with open(filepath,'rb') as infile:
            self.__dict__ = pickle.load(infile).__dict__
            self.__been_updated = False
            self.__was_loaded   = True
            print("Cluster Size Data loaded from {}".format(self.__storage_location))

        return

    def __set_storage_info(self, data_folder):

        #check if the directories /data/ and /data/sampling/ exist
        if not os.path.isdir(data_folder+"/data/"):
            os.makedirs(data_folder+"/data/")

        if not os.path.isdir(data_folder+"/data/sampling/"):
            os.makedirs(data_folder+"/data/sampling/")

        #grab the parameter identifiers from the data folder and set storage loc
        self.__storage_location  = data_folder+"data/sampling/"
        self.__storage_location += "cache.csd"

        return

    def __init_log(self):
        #make a new processed file log

        self.__been_updated = True
        self.__was_loaded   = False
        self.__processed    = []
        self.__distribution = SizeDistribution()
        self.__times_log    = []     

        #init variables to store the returned distributions
        self.__size_distribution = None
        self.__mass_weighted_sd  = None
        return

    def __process_files(self, data_folder):
        #loop over all .cl files in the folder, do analysis

        #get a sorted list of all the .cl files
        all_files = glob.glob(data_folder + '/**/*.sizes', recursive=True)
        all_files.sort()
        print("Found {} '.sizes' files in {}. Processing new files...".format(len(all_files), data_folder))

        #check if each file is already processed, if not, process it
        for traj_file in all_files:

            if traj_file not in self.__processed:
                print("Processing new file {}".format(traj_file))
                self.__process(traj_file)
                self.__processed.append(traj_file)
                self.__been_updated = True

        #save if update was performed 
        if self.__been_updated:
            self.__save()

        return

    def __parse_size_data(self, file_path):
        #read the data from size file and parse into an array

        #init temp storage for size data
        cluster_size_counts = []

        #open file and append each distribution of sizes
        with open(file_path, 'r') as f:
            for line in f:
                split_line = line.split()
                L          = len(split_line)
                frame_data = [int(val) for val in split_line[1:(L-1)]]
                cluster_size_counts.append(frame_data)

        return cluster_size_counts


    def __process(self, file_path):
        #process a single file

        #parse the data into an array
        cluster_size_counts = self.__parse_size_data(file_path)

        #make an np array with the sizes. create a distribution and add it to self
        cluster_size_counts = np.array(cluster_size_counts)
        cluster_size_distr  = SizeDistribution(cluster_size_counts)
        self.__distribution+= cluster_size_distr

        #append the number of time points in this trajectory to the log
        self.__times_log.append(cluster_size_distr.get_num_time_points())

        return

    def get_normalized_distribution(self, mass_weighted = False):
        #compute a normalized probability distribution at each frame 

        if len(self.__processed) == 0:
            return None

        #if an update has occured, recompute normalized distributions
        if self.__been_updated or self.__size_distribution is None:
            
            #non mass-weighted
            self.__size_distribution = self.__normalize_distribution()

            #mass-weighted
            self.__mass_weighted_sd  = self.__mass_weight_distribution(self.__size_distribution)


        #return the corresponding distribution
        if mass_weighted:
            return self.__mass_weighted_sd
        else:
            return self.__size_distribution

    def get_num_samples(self):

        return len(self.__processed)

    def __compute_num_samples_series(self):
        #get a time series for the number of samples summed in each time index

        #get the max number of time points and init an array for number of trajs
        max_time    = np.max(self.__times_log)
        num_samples = np.zeros(max_time, dtype=int)

        #fill the numbe rof samples array using elements of times_log
        for final_time in self.__times_log:

            num_samples[:final_time] += 1

        return num_samples

    def __normalize_distribution(self):
        #return a normalized distribution of the count data
        #first divide each time by number of samples at that time, then L1 normalize rows

        #get the cluster size counts and the time series of samples
        cluster_size_distribution = self.__distribution.get_cluster_size_data()
        num_samples = self.__compute_num_samples_series()

        #normalize by the number of samples, compute row sums, normalize by row sums
        cluster_size_distribution = cluster_size_distribution / num_samples[:, None]
        row_sums                  = np.linalg.norm(cluster_size_distribution, axis=1, ord=1)

        return (cluster_size_distribution / row_sums[:, None])

    def __mass_weight_distribution(self, size_distribution):
        #mass weight the given distribution
        #can assume index i has mass i+1 for now

        #get the weights for all sizes present in the systems
        num_states = self.__distribution.get_largest_size()
        size_range = np.arange(1, num_states+1)

        #multiply distribution by masses and renormalize
        mw_distr = size_distribution * size_range
        row_sums = np.linalg.norm(mw_distr, axis=1, ord=1)
        
        return (mw_distr / row_sums[:, None])




