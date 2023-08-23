import numpy as np
import scipy
from scipy import sparse
import sys

import warnings


class MSM:

    def __init__(self, num_states, lag=1, prune_threshold = None):

        #init the sparse matrix storage
        self.__num_states   = num_states
        self.__count_matrix = scipy.sparse.dok_matrix((num_states, num_states), dtype=int)
        self.__P            = scipy.sparse.csr_matrix((num_states, num_states), dtype=float)
        self.__row_counts   = np.zeros(num_states, dtype=int)
        self.__row_counts_uw= np.zeros(num_states, dtype=int)

        self.__lag = lag
        self.__prune_threshold = prune_threshold

        return

    def add_count(self, a, b, counts = 1):
        #increment the (a,b) entry of the count matrix

        self.__count_matrix[a,b] += counts
        return

    def finalize_counts(self, macrostate_map):
        #after all counts are added, convert to a csr matrix and compute row sums
        #use these to construct a row normalized probability transition matrix

        #do optional pruning, supress efficiency warning for sparse mat
        if self.__prune_threshold is not None:
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
            self.__compute_probability_matrix()

        return

    def __compute_probability_matrix(self):
        #use the count matrix and counts to get a normalized probability matrix

        #create a sparse matrix with reciprocal rows sums on the diagonal
        c = scipy.sparse.diags(1/self.__row_counts.ravel())

        #empty rows will get an inf. Set these to zero and log them as inactive
        find_inf = scipy.sparse.find(c > 1)[0]
        c.tocsr()
        c.data[0][find_inf] = 0
        
        #compute dot product to get PTM
        self.__P = c.dot(self.__count_matrix)
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
        for me... COnverting to coo, eliminating, then converting to
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

    def get_transition_matrix(self):

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

        #TODO - maybe have the indiviudal MSMs have this function implemented?

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