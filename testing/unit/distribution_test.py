import os
import pytest

#add the paths to the source code folder for testing
import sys
sys.path.insert(0, '../../MultiMSM/util')

import numpy as np
import sampling_data


def run_bidirectional_test(array1, array2, true_sum):
	#test that the sums are correct regardless of ordering

	d1 = sampling_data.SizeDistribution(array1)
	d2 = sampling_data.SizeDistribution(array2)

	d1 += d2
	assert(np.array_equal(true_sum, d1.get_cluster_size_data()))

	d1 = sampling_data.SizeDistribution(array2)
	d2 = sampling_data.SizeDistribution(array1)

	d1 += d2
	assert(np.array_equal(true_sum, d1.get_cluster_size_data()))

	return


def same_test():
	#test that distributions get added normally when they are both the same size

	array1 = np.array([[1,2,3],[4,5,6]])
	array2 = np.array([[0,1,0],[1,1,1]])
	true_sum = array1 + array2

	run_bidirectional_test(array1, array2, true_sum)
	print("Same Size test passed")
	return


def size_tests():
	#check that distributions add correctly when they have same time dim but different sizes

	array1 = np.array([[1,2,3],[4,5,6]])
	array2 = np.array([[5,2],[3,3]])
	true_sum = np.array([[6,4,3],[7,8,6]])

	run_bidirectional_test(array1, array2, true_sum)
	print("Bi-directional Size Tests Passed")
	return


def time_tests():
	#check that distributions add correctly when they have same size but different time disc

	array1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
	array2 = np.array([[5,2,1],[3,3,2]])
	true_sum = np.array([[6,4,4],[7,8,8],[7,8,9]])

	run_bidirectional_test(array1, array2, true_sum)
	print("Bi-directional Time Tests Passed")
	return


def size_and_time_tests():
	#check that addition works when both dims are inconsistent

	array1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
	array2 = np.array([[5,2],[3,3]])
	true_sum = np.array([[6,4,3],[7,8,6],[7,8,9]])

	run_bidirectional_test(array1, array2, true_sum)
	print("Bi-directional Full Test 1 Passed")

	array1 = np.array([[1,2,3],[4,5,6]])
	array2 = np.array([[5,2],[3,3], [5,1]])
	true_sum = np.array([[6,4,3],[7,8,6],[5,1,0]])

	run_bidirectional_test(array1, array2, true_sum)
	print("Bi-directional Full Test 2 Passed")

	return












def run_tests():
	#run a combination of tests for the distribution class

	same_test()
	size_tests()
	time_tests()
	size_and_time_tests()






if __name__ == "__main__":

	run_tests()