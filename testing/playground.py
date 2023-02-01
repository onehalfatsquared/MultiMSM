
from MultiMSM.util.discretization import Discretization
from MultiMSM.util.state import MacrostateMap
from MultiMSM.MSM.MultiMSM import Collection
from MultiMSM.MSM.MSM import MSM











if __name__ == "__main__":

    d = Discretization()
    d.set_equally_spaced(11)
    print(d.get_num_intervals())
    print(d.determine_interval(0.05))

    mm = MacrostateMap(monomer_state = 1)
    test_states = [1,2,2,2,2,2,1,5,5]
    for state in test_states:
        mm.update_maps(state)

    print(mm.state_to_index(5), mm.index_to_state(2))
    print(mm.state_to_index(4))

    m = MSM(3)
    m.add_count(0,1)
    m.add_count(0,1)
    m.add_count(0,2)
    m.add_count(1,1)
    m.add_count(1,2)
    m.add_count(2,0)
    m.add_count(2,1)
    m.add_count(2,0)
    m.add_count(2,2)

    print(m.get_count_matrix(),"\n")
    m.finalize_counts()
    print(m.get_count_matrix(),"\n")
    print(m.get_row_counts())

    print(m.get_transition_matrix(),"\n")




    C = Collection(d, mm, {'test':True})
    transitions = [(1,1,0.95),(1,2,0.95),(1,5,0.95),(1,5,0.99),(2,1,0.75),(5,1,0.44)]
    for transition in transitions:
        C.add_transition(transition[0], transition[1], transition[2])

    C.finalize_counts()

    C.print_all_transition_matrices()