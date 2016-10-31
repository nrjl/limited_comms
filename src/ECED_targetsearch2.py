import numpy as np
import random
import equivalence_class_solvers
import matplotlib.pyplot as plt
import time
import pickle
import csv
import os


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# plt.ion()
# plt.show()

# Each state (theta) is represented by a tuple (location, target_type)
n_positions = 64  # Number of possible locations
n_types = 4      # Number of possible target types

n_zones = 4

n_sims = 500     # Number of simulated tests to run
max_tests = 200   # Max tests in each simulation

write_files = True

termination_threshold = 0.02    # Max probability of error threshold

nearby_width = n_positions/n_zones/4  # 8    #

# Time stamp string
nowstr = time.strftime("%Y_%m_%d-%H_%M")
outdir = '../data/'+nowstr+'/'
ensure_dir(outdir)

# Generate all thetas using combinations of locations and types
Theta = [(pos, typ) for pos in range(n_positions) for typ in range(n_types)]

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

for pos, typ in Theta:
    zone = pos/(n_positions/n_zones)
    r[zone].add((pos, typ))

nT = len(Theta)

# These tests return the vector of likelihoods for possible returns (in this
# case binary) given a hypothesis theta and target parameter


# The location test tests a single location
class LocationTest(object):
    def __init__(self, test_pos):
        self.pos = test_pos

    def run_test(self, theta):
        # If the hypothesis position is the same as the true position, then the
        # likelihood of returning True is true_positive
        true_positive = 0.8
        if theta[0] == self.pos:
            p_true = true_positive
        else:
            p_true = 1.0-true_positive
        return [1.0-p_true, p_true]  # [p_false, p_true]


# The type test tests the target type
class TypeTest(object):
    def __init__(self, test_type):
        self.typ = test_type

    def run_test(self,theta):
        true_positive = 0.8
        if theta[1] == self.typ:
            p_true = true_positive
        else:
            p_true = 1.0-true_positive
        return [1.0-p_true, p_true]


# The nearby test is like the location test but covers a wider range (i.e likely to return true if within distance 4,
# slightly less likely to return true if within distance 8, otherwise unlikely to return true
class NearbyTest(LocationTest):
    def run_test(self, theta):
        true_positive = 0.8
        if abs(theta[0]-self.pos) < nearby_width:
            p_true = true_positive
        elif abs(theta[0]-self.pos) < nearby_width*4:
            p_true = true_positive-0.1
        else:
            p_true = 1.0-true_positive
        return [1.0-p_true, p_true]

# This generates the test for all locations and type
testclasses = []
testclasses.extend([LocationTest(pp) for pp in range(n_positions)])
testclasses.extend([NearbyTest(pp) for pp in range(n_positions)])
testclasses.extend([TypeTest(tt) for tt in range(n_types)])

# Tests actually need to be functions, so pull out functions
tests = [tc.run_test for tc in testclasses]

# Prior over possible states (uniform)
theta_prior = {t: 1.0/nT for t in Theta}

# Create all the solvers
methods = [equivalence_class_solvers.ECED,
           #equivalence_class_solvers.EC_bayes,
           equivalence_class_solvers.EC_random,
           equivalence_class_solvers.EC_US]
           #equivalence_class_solvers.EC_IG]
solver_names = ['ECED', 'Random', 'US']  # 'EC Bayes', , 'IG'

solvers = [method(Theta, r, tests, theta_prior, Theta[0]) for method in methods]
n_solvers = len(solvers)

# This contains the counts of the number of successful choices of action at each step of simulations
correct_predictions = np.zeros((max_tests, n_solvers))

# Number of steps to termination
steps_to_term = np.ones((n_sims, n_solvers), dtype='int')*max_tests
correct_at_term = np.zeros((n_sims, n_solvers), dtype='bool')

# Prediction confidence (mean MAP p(y))
map_p = np.zeros((max_tests, n_solvers))

# This is a sanity check to check how many times each test is being selected by each method
test_picks = np.zeros((len(tests), n_solvers), dtype='int')

fail_success = ['incorrectly', 'correctly']

random.seed(0)

for ii in range(n_sims):
    # For each simulation, choose a state in the first action set
    theta_true = random.sample(r[0], 1)[0]

    # Work out what the correct action is from the true state
    y_true = 0
    print "Run {0}/{1} - True theta: {2}, True y: {3}".format(ii+1, n_sims, theta_true, y_true)

    # Reset all the solvers with the true state
    for solver in solvers:
        solver.reset(theta_true)

    # Counter for number of tests and which tests are still active
    active_tests = np.ones(n_solvers, dtype='bool')
    jj = 0

    # Run solvers
    while jj < max_tests and any(active_tests):
        for ns, solver in enumerate(solvers):
            if active_tests[ns]:
                test_num,result = solver.step()
                y_predicted,map_p_y = solver.predict_y()
                correct = (y_predicted == y_true)
                if correct:
                    correct_predictions[jj, ns] += 1
                test_picks[test_num, ns] += 1
                map_p[jj, ns] += (map_p_y-map_p[jj, ns])/(ii+1)
                if 1.0 - map_p_y < termination_threshold:
                    active_tests[ns] = False
                    steps_to_term[ii, ns] = jj+1
                    correct_at_term[ii, ns] = correct
        jj += 1
    print steps_to_term[ii, :]

print test_picks
hf, ha = plt.subplots()
ha.boxplot(steps_to_term, labels=solver_names)
ha.set_xlabel('Method')
ha.set_ylabel('Observations to termination')
# ha.legend(loc=0)

print "Mean tests at termination: {0}".format(np.mean(steps_to_term, axis=0))
print "Correct at termination: {0}".format(['{:.2f}'.format(i) for i in np.sum(correct_at_term, axis=0)*100.0/n_sims])
# hpf,hpa = plt.subplots()
# hpa.bar(np.arange(n_solvers), np.sum(correct_at_term, axis=0)*100.0/n_sims)
# hpa.set_xlabel('Method')
# hpa.set_ylabel(r'Correct prediction at termination $\%$')
# hpa.legend(loc=0)


if write_files:
    with open(outdir+'predictions.pkl', 'wb') as fh:
        pickle.dump(correct_predictions, fh)
        pickle.dump(map_p, fh)

    with open(outdir+'predictions.csv', 'wb') as fh:
        csvw = csv.writer(fh, delimiter=',')
        csvw.writerow(solver_names)
        for row in correct_predictions:
            csvw.writerow(row/n_sims)

    with open(outdir+'obsterm.pkl', 'wb') as fh:
        pickle.dump(steps_to_term, fh)
        pickle.dump(correct_at_term, fh)

    with open(outdir+'obsterm.csv', 'wb') as fh:
        csvw = csv.writer(fh, delimiter=',')
        csvw.writerow(solver_names)
        for row in steps_to_term:
            csvw.writerow(row)

    with open(outdir+'setup.txt', 'wt') as fh:
        fh.write('n_positions = {0}\n'.format(n_positions))
        fh.write('n_types = {0}\n'.format(n_types))
        fh.write('n_zones = {0}\n'.format(n_zones))
        fh.write('n_sims = {0}\n'.format(n_sims))
        fh.write('max_tests = {0}\n'.format(max_tests))
        fh.write('termination_threshold = {0}\n'.format(termination_threshold))

# plt.draw()
# plt.ioff()
plt.show()

# Scrap
# solvers = {method.get_name(): method(Theta, r, tests, theta_prior, Theta[0]) for method in methods}
