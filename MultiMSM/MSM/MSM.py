import numpy as np
import scipy
from scipy import sparse
from sklearn.preprocessing import normalize

import warnings


class MSM:

    def __init__(self, num_states, lag=1):

        #init the sparse matrix storage
        self.__num_states   = num_states
        self.__count_matrix = scipy.sparse.dok_matrix((num_states, num_states), dtype=int)
        self.__P            = scipy.sparse.csr_matrix((num_states, num_states), dtype=float)
        self.__row_counts   = np.zeros(num_states, dtype=int)

        self.__lag = lag

        return

    def add_count(self, a, b, counts = 1):
        #increment the (a,b) entry of the count matrix

        self.__count_matrix[a,b] += counts
        return

    def finalize_counts(self, macrostate_map):
        #after all counts are added, convert to a csr matrix and compute row sums
        #use these to construct a row normalized probability transition matrix

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

    def get_count_matrix(self):

        return self.__count_matrix

    def get_row_counts(self):

        return self.__row_counts

    def get_transition_matrix(self):

        return self.__P

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