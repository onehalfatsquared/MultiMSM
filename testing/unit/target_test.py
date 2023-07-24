'''
    Test that the TargetState class is working as intended. 
'''

from MultiMSM.util.state import MacrostateMap
from MultiMSM.util.state import TargetStates

import unittest
from unittest.mock import patch

import pickle
import sys

#CRITICAL: If things are pickled within the util module of SAASH, 
#the following is required to unpickle outside of that project's namespace
from SAASH import util
sys.modules['util'] = util

def setupMap(database_file):
    #create a mapping from observed states to integer indexes
    #The monomer state should have index 0

    #open the pickled file
    with open(database_file, 'rb') as f:
        ref_collect = pickle.load(f)

    #extract the dictionary of states from the database
    state_dict = ref_collect.get_dict()

    #add the Monomer state to the map with index 0
    mm = MacrostateMap()

    #loop over all other states and add them
    for state in state_dict.keys():
        mm.update_maps(state)

    return mm

class TestTargetStates(unittest.TestCase):

    def test_add_indices(self):
        # Test if adding indices works correctly

        MM = setupMap("state_database.sref")
        target_states = TargetStates(MM)

        target_states.add_indices([1, 2, 3])
        self.assertEqual(target_states.get_indices(), [1, 2, 3])

        # Adding duplicate indices should not change the set
        target_states.add_indices([1, 3, 5])
        self.assertEqual(target_states.get_indices(), [1, 2, 3, 5])

    def test_get_states(self):
        # Test if getting states works correctly

        MM = setupMap("state_database.sref")
        target_states = TargetStates(MM, [1, 2, 3])

        self.assertEqual(target_states.get_states(), [MM.index_to_state(1), MM.index_to_state(2), MM.index_to_state(3)])

    @patch('builtins.input', return_value="4 5 6")
    def test_query(self, mock_input):
        # Test if querying and adding states work correctly

        MM = setupMap("state_database.sref")
        target_states = TargetStates(MM, [1, 2, 3])

        target_states.query(2)

        self.assertEqual(target_states.get_indices(), [1, 2, 3, 4, 5, 6])

    @patch('builtins.input', return_value="invalid 5 6")
    def test_invalid_input(self, mock_input):
        # Test for handling invalid input during query

        MM = setupMap("state_database.sref")
        target_states = TargetStates(MM)

        with self.assertRaises(TypeError):
            target_states.query(2)

if __name__ == '__main__':
    unittest.main()

