import numpy as np
import random
import equivalence_class_solvers
import matplotlib.pyplot as plt
import time
import pickle
import csv

# Each state (theta) is represented by a tuple (location, target_type)
n_positions = 3  # Number of possible locations
n_types = 3      # Number of possible target types

n_zones = 3

n_sims = 1000     # Number of simulated tests to run
max_tests = 40   # Max tests in each simulation

write_files = False

# Time stamp string
nowstr = time.strftime("%Y_%m_%d-%H_%M")

# Generate all thetas using combinations of locations and types
Theta = [(pos,typ) for pos in range(n_positions) for typ in range(n_types)]

# r is the function that maps states (theta) to actions (y), in this case it is an array of sets
# So r[0] is a set of states that correspond to the first action etc.
#
# # In this implementation, locations are divided into 3 equally sized zones
# # First zone is dangerous and we want to perform action 0 if target in this zone (regardless of type)
# # Second zone is controlled, and we perform action 1 if target in this zone and type 1
# # Third zone is safe, take action 2 if target in zone 2 and type 0 or zone 3 and any type
# r = [set(),set(),set()]
# for pos,typ in Theta:
#     if pos < n_positions/3: # If in zone 1, always neutralize zone 1
#         r[0].add((pos,typ))
#     elif pos < 2*n_positions/3 and typ == 1: # If in zone 2 and dangerous type (1), neutralize zone 2
#         r[1].add((pos,typ))
#     else: # Else do nothing
#         r[2].add((pos,typ))

r = [set() for i in range(n_zones)]
# for pos,typ in Theta:
#     zone = pos/(n_positions/n_zones)
#     danger = typ
#     if zone + typ < n_zones-1:
#         r[zone].add((pos,typ))
#     else:
#         r[n_zones-1].add((pos,typ))

for pos,typ in Theta:
    zone = pos/(n_positions/n_zones)
    r[zone].add((pos, typ))

nT = len(Theta)

# These tests return the vector of likelihoods for possible returns (in this 
# case binary) given a hypothesis theta and target parameter


# The location test tests a single location
class LocationTest(object):
    def __init__(self, pos):
        self.pos = pos

    def run_test(self, theta):
        # If the hypothesis position is the same as the true position, then the
        # likelihood of returning True is true_positive
        true_positive = 0.7
        if theta[0] == self.pos:
            p_true = true_positive
        else:
            p_true = 1.0-true_positive
        return [1.0-p_true, p_true]  # [p_false, p_true]


# The type test tests the target type
class TypeTest(object):
    def __init__(self, typ):
        self.typ = typ

    def run_test(self,theta):
        true_positive = 0.7
        if theta[1] == self.typ:
            p_true = true_positive
        else:
            p_true = 1.0-true_positive
        return [1.0-p_true, p_true]


# The nearby test is like the location test but covers a wider range (i.e likely to return ture if within distance 4,
# slightly less likely to return true if within distance 8, otherwise unlikely to return true
class NearbyTest(LocationTest):
    def run_test(self, theta):
        true_positive = 0.9
        if abs(theta[0]-self.pos) < 4:
            p_true = true_positive
        elif abs(theta[0]-self.pos) < 6:
            p_true = true_positive-0.1
        else:
            p_true = 1.0-true_positive
        return [1.0-p_true, p_true]

# This generates the test for all locations and type
testclasses = []
testclasses.extend([LocationTest(pp) for pp in range(n_positions)])
# testclasses.extend([NearbyTest(pp) for pp in range(n_positions)])
testclasses.extend([TypeTest(tt) for tt in range(n_types)])

# Tests actually need to be functions, so pull out functions
tests = [tc.run_test for tc in testclasses]

# Prior over possible states (uniform)
theta_prior = {t: 1.0/nT for t in Theta}

# Create all the solvers
methods = [equivalence_class_solvers.ECED,
           equivalence_class_solvers.EC_bayes,
           equivalence_class_solvers.EC_random,
           equivalence_class_solvers.EC_US,
           equivalence_class_solvers.EC_IG]
solver_names = ['ECED', 'EC Bayes', 'Random', 'US', 'IG']

solvers = [method(Theta, r, tests, theta_prior, Theta[0]) for method in methods]

# This contains the counts of the number of successful choices of action at each step of simulations
correct_predictions = np.zeros((max_tests, len(solvers)))

# Prediction confidence (mean MAP p(y))
map_p = np.zeros((max_tests, len(solvers)))

# This is a sanity check to check how many times each test is being selected by each method
test_picks = np.zeros((len(tests), len(solvers)), dtype='int')

random.seed(0)

for ii in range(n_sims):
    # For each simulation, randomly choose a true state
    theta_true = random.choice(Theta)

    # Work out what the correct action is from the true state
    for jj,rY in enumerate(r):
        if theta_true in rY:
            y_true = jj
    print "Run {0}/{1} - True theta: {2}, True y: {3}".format(ii+1, n_sims, theta_true, y_true)

    # Reset all the solvers with the true state
    for solver in solvers:
        solver.reset(theta_true)

    # Run solvers
    for jj in range(max_tests):
        for ns, solver in enumerate(solvers):
            test_num,result = solver.step()
            # print "Y posterior: {0}".format(solver.p_Y)
            y_predicted,map_p_y = solver.predict_y()
            if y_predicted == y_true:
                correct_predictions[jj, ns] += 1
            test_picks[test_num, ns] += 1
            map_p[jj, ns] += (map_p_y-map_p[jj, ns])/(ii+1)

print test_picks
hf,ha = plt.subplots()
for ii in range(len(solvers)):
    ha.plot(correct_predictions[:, ii]/n_sims, label=solver_names[ii])
ha.set_xlabel('Number of tests executed')
ha.set_ylabel('Correct action predictions')
ha.legend(loc=0)

hpf,hpa = plt.subplots()
for ii in range(len(solvers)):
    hpa.plot(1.0-map_p[:, ii], label=solver_names[ii])
hpa.set_xlabel('Number of tests executed')
hpa.set_ylabel(r'Error probability ($1-\max_{y}p(y|\psi)$)')
hpa.legend(loc=0)

if write_files:
    with open('../data/'+nowstr+'.pkl', 'wb') as fh:
        pickle.dump(correct_predictions, fh)
        pickle.dump(map_p, fh)

    with open('../data/'+nowstr+'.csv', 'wb') as fh:
        csvw = csv.writer(fh, delimiter=',')
        csvw.writerow(solver_names)
        for row in correct_predictions:
            csvw.writerow(row/n_sims)

plt.show()


# JUNK
# def location_test(theta,position):
#    if theta[0] == position:
#        p_true = 0.8
#    else:
#        p_true = 0.2
#    return [1.0-p_true, p_true] #[p_false,p_true]
#    
# def type_test(theta,typ):
#     if theta[1] == typ:
#         p_true = 0.7
#     else:
#         p_true = 0.3
#     return [1.0-p_true, p_true]

# tests = [lambda(t): location_test(t,position=pos) for pos in range(n_positions)]
# tests.extend([lambda(t):type_test(t,typ=ty) for ty in range(n_types)])
# tests = [lambda(t): location_test(t,position=0),
#          lambda(t):location_test(t,position=1),
#          lambda(t):location_test(t,position=2),
#          lambda(t):type_test(t,typ=0),
#          lambda(t):type_test(t,typ=1)]
# Theta = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
# r = [{Theta[0],Theta[1]},{Theta[2]},{Theta[3],Theta[4],Theta[5]}]
# tests = [lambda(t): location_test(t,position=0),
#          lambda(t):location_test(t,position=1),
#          lambda(t):location_test(t,position=2),
#          lambda(t):type_test(t,typ=0),
#          lambda(t):type_test(t,typ=1)]
