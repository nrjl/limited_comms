import numpy as np
import random
import equivalence_class_solvers
import matplotlib.pyplot as plt

# 3 locations, 2 target types
n_positions = 30
n_types = 2

Theta = [(pos,typ) for pos in range(n_positions) for typ in range(n_types)]

r = [set(),set(),set()]
for pos,typ in Theta:
    if pos < n_positions/3: # If in the first third of the locations, always neutralize
        r[0].add((pos,typ))
    elif pos < 2*n_positions/3 and typ == 1: # If in second third and dangerous, neutralize
        r[1].add((pos,typ))
    else: # Else do nothing
        r[2].add((pos,typ))

nT = len(Theta)

# These tests return the vector of likelihoods for possible returns (in this 
# case binary) given a hypothesis theta and target parameter
class LocationTest(object):
    def __init__(self,pos):
        self.pos = pos
    def run_test(self,theta):
    # If the hypothesis position is the same as the true position, then the
    # likelihood of returning True is 0.8
        true_pos = 0.99
        if theta[0] == self.pos:
            p_true = true_pos
        else:
            p_true = 1.0-true_pos
        return [1.0-p_true, p_true] #[p_false,p_true] 
        
class TypeTest(object):
    def __init__(self,typ):
        self.typ = typ
    def run_test(self,theta):
        true_pos = 0.7
        if theta[1] == self.typ:
            p_true = true_pos
        else:
            p_true = 1.0-true_pos
        return [1.0-p_true, p_true]
        
#class NearbyTest(LocationTest):
#    def run_test(self, theta):
#        if abs(theta[0]-self.pos) < 3:
#            p_true = 0.8
#        return p_true

testclasses = [LocationTest(pp) for pp in range(n_positions)]
testclasses.extend([TypeTest(tt) for tt in range(n_types)])
tests = [tc.run_test for tc in testclasses]

theta_prior = {t:1.0/nT for t in Theta}

n_sims = 100
max_tests = 30

methods = [equivalence_class_solvers.ECED, 
            equivalence_class_solvers.EC_bayes,
            equivalence_class_solvers.EC_random,
            equivalence_class_solvers.EC_US,
            equivalence_class_solvers.EC_IG]
solver_names = ['ECED', 'EC-Bayes', 'Random', 'US', 'IG']

solvers = [method(Theta,r,tests,theta_prior,Theta[0]) for method in methods]
correct_predictions = np.zeros((max_tests,len(solvers)))

random.seed(0)

for ii in range(n_sims):
    theta_true = random.choice(Theta)
    for jj,rY in enumerate(r):
        if theta_true in rY:
            y_true = jj
    print "Run {0}/{1} - True theta: {2}, True y: {3}".format(ii+1,n_sims,theta_true, y_true)
    for solver in solvers:
        solver.reset(theta_true)
    
    for jj in range(max_tests):
        for ns,solver in enumerate(solvers):
            solver.step()
            # print "Y posterior: {0}".format(solver.p_Y)
            if solver.predict_y() == y_true:
                correct_predictions[jj,ns] += 1

hf,ha = plt.subplots()
for ii in range(len(solvers)):
    ha.plot(correct_predictions[:,ii]/n_sims,label=solver_names[ii])     
ha.set_xlabel('Number of tests executed')
ha.set_ylabel('Correct action predictions')
ha.legend()
hf.show()



# JUNK
# def location_test(theta,position):
#    if theta[0] == position:
#        p_true = 0.8
#    else:
#        p_true = 0.2
#    return [1.0-p_true, p_true] #[p_false,p_true]
#    
#def type_test(theta,typ):
#    if theta[1] == typ:
#        p_true = 0.7
#    else:
#        p_true = 0.3
#    return [1.0-p_true, p_true]
#
#tests = [lambda(t): location_test(t,position=pos) for pos in range(n_positions)]
#tests.extend([lambda(t):type_test(t,typ=ty) for ty in range(n_types)])
#tests = [lambda(t): location_test(t,position=0), 
#        lambda(t):location_test(t,position=1),
#        lambda(t):location_test(t,position=2), 
#        lambda(t):type_test(t,typ=0),       
#        lambda(t):type_test(t,typ=1)]
#Theta = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
#r = [{Theta[0],Theta[1]},{Theta[2]},{Theta[3],Theta[4],Theta[5]}]
#tests = [lambda(t): location_test(t,position=0), 
#        lambda(t):location_test(t,position=1),
#        lambda(t):location_test(t,position=2), 
#        lambda(t):type_test(t,typ=0),       
#        lambda(t):type_test(t,typ=1)]
