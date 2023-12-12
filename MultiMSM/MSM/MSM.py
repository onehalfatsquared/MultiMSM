import numpy as np
import scipy
from scipy import sparse
import sys
import copy

import matplotlib.pyplot as plt

import warnings


class MSM:

    def __init__(self, num_states, lag=1, prune_threshold = None):

        #init the sparse matrix storage
        self.__num_states    = num_states
        self.__count_matrix  = scipy.sparse.dok_matrix((num_states, num_states), dtype=int)
        self.__P             = scipy.sparse.csr_matrix((num_states, num_states), dtype=float)
        self.__row_counts    = np.zeros(num_states, dtype=int)
        self.__row_counts_uw = np.zeros(num_states, dtype=int)

        self.__macrostate_map = None

        self.__clusterized_P = None

        self.__reduced_P     = None
        self.__reduced_map   = None
        self.__reduced_eq    = None

        self.__lag = lag
        self.__prune_threshold = prune_threshold

        return

    def add_count(self, a, b, counts = 1):
        #increment the (a,b) entry of the count matrix

        self.__count_matrix[a,b] += counts
        return
    
    def finalize_counts(self, macrostate_map, weighted=True):
        #after all counts are added, convert to a csr matrix and compute row sums
        #use these to construct a row normalized probability transition matrix

        self.__macrostate_map = macrostate_map

        #do optional pruning, supress efficiency warning for sparse mat
        if self.__prune_threshold is not None and self.__prune_threshold > 1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore') #sparse efficiency warning, cant be avoided
                self.__count_matrix[self.__count_matrix < self.__prune_threshold] = 0
        
        #store the unweighted row counts before performing the weighting
        self.__row_counts_uw = np.asarray(self.__count_matrix.sum(axis=1)).squeeze()

        #weight the count matrix to correctly represent mass-weighted dynamics
        self.__perform_weighting(macrostate_map)

        #convert sparse matrix to csr for more efficient arithmetic. get row sums. 
        self.__count_matrix = self.__count_matrix.tocsr()
        self.__row_counts   = np.asarray(self.__count_matrix.sum(axis=1)).squeeze()

        #normalize the rows to construct a transition matrix
        with warnings.catch_warnings():
            warnings.simplefilter('ignore') #sparse efficiency warning, cant be avoided?
            self.__compute_probability_matrix(weighted)

        return

    def __compute_probability_matrix(self, weighted=True):
        #use the count matrix and counts to get a normalized probability matrix

        #create a sparse matrix with reciprocal rows sums on the diagonal
        if weighted:
            c = scipy.sparse.diags(1/self.__row_counts.ravel())
        else:
            c = scipy.sparse.diags(1/self.__row_counts_uw.ravel())

        #empty rows will get an inf. Set these to zero and log them as inactive
        find_inf = scipy.sparse.find(c > 1)[0]
        c.tocsr()
        c.data[0][find_inf] = 0
        
        #compute dot product to get PTM
        if weighted:
            self.__P = c.dot(self.__count_matrix)
        else:
            self.__P = c.dot(self.__unweighted_counts)
        diagonal = self.__P.diagonal()

        #check which rows have no entries, set diag to 1 there
        for i in range(self.__num_states):
            r = self.__P.getrow(i)
            if (r.sum() < 1e-4):
                diagonal[i] = 1.0

        self.__P.setdiag(diagonal)

        return

    def get_lag(self):

        return self.__lag

    def set_count_matrix(self, count_matrix, macrostate_map):

        self.__count_matrix = count_matrix
        self.finalize_counts(macrostate_map)

        return
    
    def set_row_prob(self, row_idx, distribution):
        #replace the row in the transition matrix with the given distr.

        #replace the row
        self._set_row_csr(self.__P, row_idx, distribution)

        '''
        Eliminate the zeros that may have been added in a swap. 
        Note: eliminate zeros does not seem to work on csr matrix
        for me... Converting to coo, eliminating, then converting to
        csr seems to work.
        '''
        self.__P = self.__P.tocoo()
        self.__P.eliminate_zeros()
        self.__P = self.__P.tocsr()

        return

    def _set_row_csr(self, A, row_idx, new_row):
        #change the row of a csr matrix to a new row

        assert sparse.isspmatrix_csr(A), 'A shall be a csr_matrix'
        assert row_idx < A.shape[0], \
                'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
                .format(row_idx, A.shape[0])
        try:
            N_elements_new_row = len(new_row)
        except TypeError:
            msg = 'Argument new_row shall be a list or numpy array, is now a {0}'\
            .format(type(new_row))
            raise AssertionError(msg)
        N_cols = A.shape[1]
        assert N_cols == N_elements_new_row, \
                'The number of elements in new row ({0}) must be equal to ' \
                'the number of columns in matrix A ({1})' \
                .format(N_elements_new_row, N_cols)

        idx_start_row = A.indptr[row_idx]
        idx_end_row = A.indptr[row_idx + 1]
        additional_nnz = N_cols - (idx_end_row - idx_start_row)

        A.data = np.r_[A.data[:idx_start_row], new_row, A.data[idx_end_row:]]
        A.indices = np.r_[A.indices[:idx_start_row], np.arange(N_cols), A.indices[idx_end_row:]]
        A.indptr = np.r_[A.indptr[:row_idx + 1], A.indptr[(row_idx + 1):] + additional_nnz]
        return
    
    def get_count_matrix(self):

        return self.__count_matrix.copy()

    def get_row_counts(self, weighted = False):

        if weighted:
            return self.__row_counts.copy()

        return self.__row_counts_uw

    def get_transition_matrix(self, clusterized=False, remove_absorbing = False):

        if clusterized and remove_absorbing:
            if self.__clusterized_P is not None and self.__reduced_P is not None:
                return self.__reduced_P
            else:
                if self.__clusterized_P is None:
                    warning ="Clusterized transition matrix was requested, but has not "
                    warning+="been computed. Call self.clusterize_matrix() first."
                    raise RuntimeError(warning)
                if self.__reduced_P is None:
                    warning = "Reduced transition matrix was requested, but has not "
                    warning+= "been computed. Call self.remove_absorbing() first."
                    raise RuntimeError(warning)


        elif clusterized:
            if self.__clusterized_P is not None:
                return self.__clusterized_P.copy()
            else:
                warning = "Clusterized transition matrix was requested, but has not "
                warning+= "been computed. Call self.clusterize_matrix() first."
                raise RuntimeError(warning)
            
        elif remove_absorbing:
            if self.__reduced_P is not None:
                return self.__reduced_P.copy()
            else:
                warning = "Reduced transition matrix was requested, but has not "
                warning+= "been computed. Call self.remove_absorbing() first."
                raise RuntimeError(warning)
            
        return self.__P.copy()

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
        p[0, :]    = p0

        #solve the FKE, grabbing the relevant transition matrix each iteration
        TM = self.get_transition_matrix()
        for t in range(T):

            #update the probabilities 1 step in future
            p[t+1, :] = p[t, :] * TM

        return p

    def __perform_weighting(self, macrostate_map):
        ''' 
        Multiply the count matrix elementwise by min(size(i),size(j)). 
        Converts from a cluster perspective (SAASH) to particle perspective,
        needed to correctly characterize mass-weighted dynamics. 
        '''

        self.__unweighted_counts = self.get_count_matrix()

        #get the nonzero indices and number of nonzero elements
        nzR, nzC = self.__count_matrix.nonzero()
        nnz = len(nzR)

        #loop over each non-zero, multiply by proper weighting
        for i in range(nnz):

            #grab the row and column index of the current entry
            row_index = nzR[i]
            col_index = nzC[i]

            #convert these state indices to States and grab their size
            row_state_size = macrostate_map.index_to_state(row_index).get_size()
            col_state_size = macrostate_map.index_to_state(col_index).get_size()

            #multiply C[r][c] by the weighting min(r_size, c_size)
            W = min(row_state_size, col_state_size)
            self.__count_matrix[row_index, col_index] *= W

        return
    
    def __get_largest_row_prob(self, row, subset = None):
        #determine the index and value of the largest probability in the row
        #if subset is given, only consider elements in this set

        #construct full list of states if no subset is given
        if subset is None:
            subset = range(self.__num_states)

        #init values for maximum
        max_id = -1
        max_val= 0

        #loop over the given indices to find the max
        for index in subset:
            new_val = self.__P[row, index]
            if new_val > max_val:
                max_val = new_val
                max_id  = index

        return max_id, max_val
    
    def __cluster_modify(self, state_id, csize, mod_size, macrostate_map):
        '''
        Looks at row of transition matrix corresponding to state_id. 
        Determines all entries with size mod_size with nonzero transition probability
        and zeros them out, moving the probability to the most likely state with size
        csize-mod_size. 
        '''

        #get all structures with size mod_size
        to_remove = macrostate_map.filter_by_size(mod_size)

        #remove all the probability to transition to these states and sum it
        removed_prob = 0
        for removed_state in to_remove:
            removed_prob += self.__clusterized_P[state_id, removed_state]
            self.__clusterized_P[state_id, removed_state] = 0

        #determine most likely csize-mod_size structure to put prob into
        possible_ids = macrostate_map.filter_by_size(csize-mod_size)
        max_id, max_val = self.__get_largest_row_prob(state_id, possible_ids)

        #modify the maximum, and handle case where no max is found
        if max_id > 0:
            self.__clusterized_P[state_id, max_id] = max_val + removed_prob
        else:
            self.__clusterized_P[state_id, state_id] += removed_prob

        return
    
    def clusterize_matrix_v2(self, macrostate_map, cluster_size_threshold = 3, 
                             recompute = False):
        #try clusterize again, but remove probability for all structures < size/2

        #dont recompute the clusterized matrix, it is always the same for same thresh
        if self.__clusterized_P is not None and not recompute:
            return

        #store a copy of the transition matrix
        self.__clusterized_P = self.get_transition_matrix()

        #loop over all states, modifying each row above the size threshold
        for state_id in range(self.__num_states):
            #get the size of this state and check if modification is needed
            csize = macrostate_map.index_to_state(state_id).get_size()
            if csize >= cluster_size_threshold:
                
                #get list of sizes strictly less than half the cluster size
                sizes_to_modify = [i for i in range(1,(csize+1)//2)]
                # sizes_to_modify = [i for i in range(1,csize-1)]

                #loop over each size to make modification
                for mod_size in sizes_to_modify:
                    self.__cluster_modify(state_id, csize, mod_size, macrostate_map)

        return
    
    def clusterize_matrix(self, macrostate_map, cluster_size_threshold = 3,
                          recompute = False):
        '''
        Attempt to make the transition matrix amenable to cluster analysis by removing
        the ability for large intermediates to transition to monomers. 

        For each row corresponding to a large enough intermediate (with cutoff size as 
        an optional argument), set the (id,0) entry to 0 and move its probability to 
        the most likely n-1 mer. Store result in a new transition matrix. 
        '''

        #dont recompute the clusterized matrix, it is always the same for same thresh
        if self.__clusterized_P is not None and not recompute:
            return

        #store a copy of the transition matrix
        self.__clusterized_P = self.get_transition_matrix()

        #loop over all states, modifying each row above the size threshold
        for state_id in range(self.__num_states):

            #check if there is a nonzero monomerization probability. if not, skip
            prob_monomer = self.__clusterized_P[state_id, 0]
            if prob_monomer < 1e-4:
                continue

            #get the size of this state and check if modification is needed
            csize = macrostate_map.index_to_state(state_id).get_size()
            if csize >= cluster_size_threshold:

                # print(state_id, csize, self.__clusterized_P[state_id, 0])

                #determine the most likely n-1 size state
                nm1_ids = macrostate_map.filter_by_size(csize-1)
                max_id, max_val = self.__get_largest_row_prob(state_id, nm1_ids)

                # print(max_id, max_val)

                #if a maximum n-1 mer is found, move the monomerization prob there
                if max_id > 0:
                    self.__clusterized_P[state_id, max_id] = max_val + prob_monomer
                    self.__clusterized_P[state_id, 0]      = 0
                else:
                    self.__clusterized_P[state_id, state_id] += prob_monomer
                    self.__clusterized_P[state_id, 0]         = 0

        return

    def compute_committor(self, initial_indices, target_indices):
        '''
        Solve for the committor function for transitioning from an intitial set of
        states to a target set of states, specified by integer indices or a list of 
        integer indices. 

        The linear system is 
            Lq = 0, q(initial) = 0, q(final) = 1, 
        where L = P-I is the generator matrix (transition matrix minus identity)
        This can instead be posed as L'q = b, where L' is L but with rows corresponding 
        to initial or target states set to the corresponding unit vector, and b(i) = 1
        if i is in the target set and b(i) = 0 otherwise. This method is used here. 

        It is not unusual for these systems to be very poorly conditioned if not 
        entirely singular, so we cannot use the standard linear solver. Instead, 
        we compute a least squares solution and zero out any negative entries. 

        '''

        #convert int indices to list for generality
        if not isinstance(initial_indices, list):
            initial_indices = [initial_indices]
        if not isinstance(target_indices, list):
            target_indices = [target_indices]

        #initialize the augmented transition matrix and RHS vector
        if self.__clusterized_P is None:
            warning = "Warning: No clusterized transition matrix was found. Using the "
            warning+= "standard transition matrix, which may result in a biased "
            warning+= "committor due to monomerization of large intermediates. "
            warning+= "Call MSM.clusterize_matrix() for unbiased result. "
            print(warning)
            TM = self.get_transition_matrix().todense()
        else:
            TM = self.__clusterized_P.todense()
        b  = np.zeros(self.__num_states, dtype=float)

        #subtract off the identity to get the generator matrix
        L = TM - np.eye(self.__num_states)

        #loop over the initial indices to make states absorbing
        for index in initial_indices:
            for j in range(self.__num_states):
                if j == index:
                    L[index, j] = 1.0
                else:
                    L[index, j] = 0.0

        #loop over the target indices to make states absorbing, augment RHS
        for index in target_indices:
            for j in range(self.__num_states):
                if j == index:
                    L[index, j] = 1.0
                else:
                    L[index, j] = 0.0
            b[index] = 1.0

        #solve the linear system Lx = b

        #pseudo-inverse method (slow)
        # B = scipy.linalg.pinv(L)
        # x = scipy.dot(B, b)

        #least squares (fast, same soln as psuedoinv)
        x = scipy.linalg.lstsq(L,b)[0]

        #zero out negative entries and cap greater than 1 entries
        x[x<0] = 0.0
        x[x>1.0] = 1.0

        return x
    
    def compute_critical_nucleus(self, initial_indices, target_indices, macrostate_map, 
                                 max_size = None):
        '''
            Gathers and returns information about the critical nucleus size. 

            First computes the committor for the transition between the specified
            initial and final states. The critical nucleus is then probed in a few ways.

            1) Identify the committor element closest to 0.5 and return the index/
            state. 

            2) Construct a vector of the maximal committor value as a function of 
            intermediate size. 

            3) Determine the first intermediate size in 2) with committor > 0.5. 

        '''

        #first, compute the committor
        q = self.compute_committor(initial_indices, target_indices)

        #identify element closest to 0.5, get the index and state
        diff = np.abs(q-0.5)
        minE = np.argmin(diff)
        crit_state = macrostate_map.index_to_state(minE)
        out1 = (minE, crit_state)

        #if no max_size is given, determine it from the map
        if max_size is None:
            max_size = 0
            for index in range(self.__num_states):
                this_size = macrostate_map.index_to_state.get_size()
                if this_size > max_size:
                    max_size = this_size

        #loop over all sizes and determine max committor value
        max_commit = np.zeros(max_size, dtype=float)
        for isize in range(max_size):
            max_q = 0
            max_index = -1
            indices = macrostate_map.filter_by_size(isize+1)
            for index in indices:
                this_q = q[index]
                if this_q > max_q:
                    max_q = this_q
                    max_index = index
            max_commit[isize] = max_q

        #determine the first index greater than 0.5
        crit_size = next(i for i, x in enumerate(max_commit) if x >= 0.5)+1

        #return all 3 measures
        return out1, max_commit, crit_size
    

    def find_communication_classes(self, transition_matrix):
        #determine communication classes by computing strongly connected component

        _, labels = scipy.sparse.csgraph.connected_components(transition_matrix, directed=True, connection='strong')
        classes = [set(np.where(labels == label)[0]) for label in range(max(labels) + 1)]
        return classes

    
    def __determine_absorbing(self, TM, absorb_tol, manually_remove = []):
        '''
        Returns an array containing all indices of absorbing and near-absorbing states. 

        Absorbing states are computed as all states that are not part of the largest
        strongly connected component of the transition matrix graph. 

        Near-absorbing states are defined as those with diagonal entries greater than
        the provided tolerance, typically larger than 0.99. 

        After these states are identified, we further iteratively check that there are
        no rows that sum to 0 after removing the identified states. If they do sum to 0,
        we remove these states as well. 
        '''

        #set max number of iterations based on number of states
        max_iter = self.__num_states

        #determine the largest strongly connected component
        cc = self.find_communication_classes(TM)
        recurrent_set = max(cc, key=len)
        absorbing_indices = [i for i in range(max_iter) if i not in recurrent_set]
        

        #loop over diagonal entries to remove. Never remove monomer (index 0)
        for i in range(self.__num_states):
            diag = TM[i,i]
            if diag > absorb_tol and i not in absorbing_indices and i > 0:
                absorbing_indices.append(i)

        #add states to manually remove
        for index in manually_remove:
            if index not in absorbing_indices:
                absorbing_indices.append(index)

        #perform iterative process 
        for iter in range(max_iter):

            num_ads_begin = len(absorbing_indices)

            non_absorbing_indices = [i for i in range(self.__num_states) if i not in absorbing_indices]

            for row_num in non_absorbing_indices:
                row = TM[row_num, :]
                new_row = np.array([row[0,i] for i in range(self.__num_states) if i not in absorbing_indices])
                S = np.sum(new_row)
                if S < 1e-6:
                    absorbing_indices.append(row_num)

            num_ads_end = len(absorbing_indices)

            if num_ads_begin == num_ads_end:
                break

        return absorbing_indices

    
    def remove_absorbing(self, macrostate_map, absorb_tol = 0.991, clusterized=False,
                         manually_remove = []):
        '''
        Remove the absorbing states from the transition matrix, ensuring to 
        re-normalize any row that no longer sums to 1. Stores this matrix in 
        self.__reduced_P. 

        Also returns a modified macrostate map, with the absorbing states removed
        and all indices shifted to match the new matrix. 
        '''

        #first, get the base transition matrix
        TM = self.get_transition_matrix(clusterized=clusterized).todense()

        #gather a list of absorbing states. sort it from largest to smallest
        absorbing_indices = self.__determine_absorbing(TM, absorb_tol, manually_remove)
        absorbing_indices = -np.sort(-np.array(absorbing_indices))

        # for id in absorbing_indices:
        #     print(id, macrostate_map.index_to_state(id))

        #remove these indices from the transition matrix, rows and columns, and new map
        new_map = copy.deepcopy(macrostate_map)
        for index in absorbing_indices:
            new_map.remove_entry(index)
            TM = np.delete(TM, index, 0)
            TM = np.delete(TM, index, 1)

        #renormalize each row
        for row_num in range(new_map.get_num_states()):
            row = TM[row_num,:]
            row_sum = np.sum(row)
            TM[row_num,:] = (row/row_sum)

        #make the new matrix sparse and save it to the class instance
        self.__reduced_P = scipy.sparse.csr_matrix(TM)
        self.__reduced_map = new_map

        return new_map
    
    def compute_equilibrium_distribution(self, num_states = None, remove_absorbing = 
                                         False, clusterized = False):
        #compute, store, and return the equilibrium distribution of the MSM
        
        #get the appropriate transition matrix wrt absorbing states
        TM = self.get_transition_matrix(remove_absorbing=remove_absorbing,
                                        clusterized=clusterized)

        #if num states is None, use the full transition matrix values
        if num_states is None:
            num_states = self.__num_states

        #try performing many matrix multiplications
        for i in range(40):
            TM = (TM * TM)

        #multiply by the monomer vector
        p     = np.zeros(num_states, dtype=float)
        p[0] = 1
        p_eq = p * TM

        self.__reduced_eq = p_eq
        return p_eq
    
    def compute_fe_vs_size(self, max_size, remove_absorbing = True, plot=False,
                           min_fe = False):
        #make plot of free energy vs intermediate size

        #assume that absorbing states have been removed and eq computed, for now

        #convert equilibrium distribution to free energies
        free_energies = -np.log(self.__reduced_eq) 

        if remove_absorbing:
            map = self.__reduced_map
        else:
            map = self.__macrostate_map

        if (min_fe):

            #make array to store minimal free energies
            free_energy_minima = np.ones(max_size, dtype=float) * 10000

            #loop over each element, get the structure size, determine minima
            for i in range(len(free_energies)):

                fe = free_energies[i]
                csize = map.index_to_state(i).get_size()
                if csize <= max_size:
                    if fe < free_energy_minima[csize-1]:
                        free_energy_minima[csize-1] = fe

        else:

            #make array to store summed free energies
            energy_sums = np.zeros(max_size, dtype=float)

            #loop over each element, get the structure size, add fe to array
            for i in range(len(self.__reduced_eq)):

                e = self.__reduced_eq[i]
                csize = map.index_to_state(i).get_size()
                if csize <= max_size:
                    energy_sums[csize-1] += e

                free_energy_minima = -np.log(energy_sums)


        #make plot of free energy vs intermediate size
        if plot:
            plt.plot(range(1,max_size+1), free_energy_minima, linewidth=2)
            plt.xlabel("Intermediate Size", fontsize=18)
            plt.ylabel("Free Energy (kT)", fontsize=18)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.show()

        return free_energy_minima


    

