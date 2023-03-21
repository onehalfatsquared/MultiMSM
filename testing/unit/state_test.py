import os
import pytest

#add the paths to the source code folder for testing
import sys
sys.path.insert(0, '../../MultiMSM')

import numpy as np

from SAASH.util.state import State
from util import state

monomer = state.Monomer(State(1,{}),0)

def test_filtering():
	#make sure filtering by size returns all desired states

	#define map and a set of example states
	MM = state.MacrostateMap(monomer)
	two_states = [State(2,{'bonds':2}), State(2,{'bonds':3})]
	three_states = [State(3,{'bonds':2}), State(3,{'bonds':3}), State(3,{'bonds':4})]
	extra = [State(4,{}), State(7,{}),State(7,{'bonds':6}), monomer.get_state()]
	all_states = two_states + extra + three_states

	#add states to map
	for test_state in all_states:
		MM.update_maps(test_state)

	#filter for 2
	two_filtered = MM.filter_by_size(2)
	two_indexed  = [MM.state_to_index(x) for x in two_states]
	assert(set(two_filtered) == set(two_indexed))

	#filter for 3
	three_filtered = MM.filter_by_size(3)
	three_indexed  = [MM.state_to_index(x) for x in three_states]
	assert(set(three_filtered) == set(three_indexed))

	#filter for 2 and 3
	two_three_filtered = MM.filter_by_size([2,3])
	two_three_indexed  = two_indexed + three_indexed
	assert(set(two_three_filtered) == set(two_three_indexed))

	#filter for 1
	one_filtered = MM.filter_by_size(1)
	one_indexed  = [0]
	assert(set(one_filtered) == set(one_indexed))

	print("Filtering Tests Passed")
	return



if __name__ == "__main__":

	test_filtering()

