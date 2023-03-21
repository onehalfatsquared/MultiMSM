import os
import pytest

#add the paths to the source code folder for testing
import sys
sys.path.insert(0, '../../MultiMSM')

import numpy as np

from SAASH.util.state import State
from MultiMSM.util.state import MacrostateMap
from MultiMSM.util.state import Monomer
from MultiMSM.util.discretization import Discretization
from MultiMSM.MSM.MultiMSM import Collection


def cache_test():
	#create a MultiMSM with fake data, cache it, and ensure the reconstruction is the same

	#remove "test.cache" if it exists"
	if os.path.isfile("test.cache"):
		os.remove("test.cache")
            
	#do all the setup for a 10 state chain over 4 intervals
	indices = [0,0.25,0.6,0.9,1]
	D = Discretization(indices)
	L = len(D)
	MM = MacrostateMap(Monomer(State(1,{}),0))
	states = [State(i,{}) for i in range(1,11)]
	for s in states:
		MM.update_maps(s)
	C = Collection(D,MM)

	#generate some fake data to fill the MSM
	N = 4000
	for i in range(N):

		index = (i % 4) 
		state1 = int(i*i) % 10
		state2 = int(i*(2+i+np.sin(i))) % 10
		frac = indices[index] + (i % 10) / 101 + 1e-6

		#add the counts with cache on
		C._Collection__add_transition(states[state1], states[state2], frac, cache=True)

	C.finalize_counts()

	#save the cache
	C._Collection__save_cache("test.pkl")

	#create a new collection using the same stuff, and load from cache
	C2 = Collection(D,MM)
	C2._Collection__load_from_cache("test.pkl")
	C2.finalize_counts()

	#delete the cache file for next time the test is done
	if os.path.isfile("test.cache"):
		os.remove("test.cache")

	#compare the count matrices in each bin
	for i in range(1,L+1):
		counts1 = C.get_count_matrix(i)
		counts2 = C2.get_count_matrix(i)

		assert(np.allclose(counts1.A, counts2.A))

	print("Cache test passed")
	return




if __name__ == "__main__":

	cache_test()