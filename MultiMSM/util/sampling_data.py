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
        # p_identity               = data_folder.split("/")[-2]
        # self.__storage_location += p_identity + ".pkl"
        self.__storage_location += "cache.csd"

        return

    def __init_log(self):
        #make a new processed file log

        self.__largest_size      = 0
        self.__been_updated      = True
        self.__was_loaded        = False
        self.__processed         = []
        self.__summed_sizes      = None
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

    def __process(self, file_path):
        #process a single file

        #init temp storage for size data
        cluster_size_counts = []

        #open file and append each distribution of sizes
        with open(file_path, 'r') as f:
            for line in f:
                split_line = line.split()
                L          = len(split_line)
                frame_data = [int(val) for val in split_line[1:(L-1)]]
                cluster_size_counts.append(frame_data)

        #make an np array with the sizes. 
        cluster_size_counts = np.array(cluster_size_counts)

        #if sum does not exist, init it as counts. if it does, add to sum
        if self.__summed_sizes is None:

            self.__summed_sizes  = np.zeros(cluster_size_counts.shape, dtype=int)
            self.__largest_size  = cluster_size_counts.shape[1]
            self.__summed_sizes += cluster_size_counts

        else:

            #check if the dimensions of new size counts match storage
            if cluster_size_counts.shape[1] > self.__largest_size:

                #ignore the larger sized structures measured
                cluster_size_counts = cluster_size_counts[:,:self.__largest_size]

            elif cluster_size_counts.shape[1] < self.__largest_size:

                #change largest size to accomdate a smaller max size
                self.__largest_size = cluster_size_counts.shape[1]
                self.__summed_sizes = self.__summed_sizes[:,:self.__largest_size]

            #add the new counts
            self.__summed_sizes += cluster_size_counts

        return

    def get_normalized_distribution(self, mass_weighted = False):
        #compute a normalized probability distribution at each frame 

        if len(self.__processed) == 0:
            return None

        #if an update has occured, recompute normalized distributions
        if self.__been_updated or self.__size_distribution is None:
            
            #non mass weighted
            row_sums = np.linalg.norm(self.__summed_sizes, axis=1, ord=1)
            self.__size_distribution = self.__summed_sizes / row_sums[:, None]

            #mass weighted - TODO - will eventually need fixing to use size of state
            num_states = self.__summed_sizes.shape[1]
            size_range = np.arange(1, num_states+1)
            summed_sizes_mw = self.__summed_sizes * size_range
            row_sums = np.linalg.norm(summed_sizes_mw, axis=1, ord=1)
            self.__mass_weighted_sd = summed_sizes_mw / row_sums[:, None]


        #return the corresponding distribution
        if mass_weighted:
            return self.__mass_weighted_sd
        else:
            return self.__size_distribution

    def get_num_samples(self):

        return len(self.__processed)

