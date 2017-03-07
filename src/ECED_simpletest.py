import equivalence_class_solvers

# Based on the simple test example in Sec. 3 EC^2-Bayes:
# Chen, Hassani, Krause. "Near-optimal Bayesian Active Learning
# with Correlated and Noisy Tests." https://arxiv.org/abs/1605.07334

# Root causes:
Theta = [1,2,3]

# Equivalence classes:
r = [set(Theta[0:2]), set([Theta[2]])]


class SimpleTest(object):
    def __init__(self, epsilon=0.05, target_theta = 1, cost=1.0):
        self.n_outcomes = 2
        self.epsilon = epsilon
        self.target_theta = target_theta
        self.cost = cost
    
    def outcome_likelihood(self, theta):
        if theta == self.target_theta:
            return [1.0-self.epsilon, self.epsilon]
        else:
            return [self.epsilon, 1.0-self.epsilon]

# Example from paper:
#theta_prior = {1:.2, 2:.4, 3:.4}
#tests = [SimpleTest(0.5), SimpleTest(0.0)]

# Example to sanity test methods:
theta_prior = {1: 0.4, 2: 0.4, 3: 0.2}
tests = [SimpleTest(0.5, target_theta=1), 
        SimpleTest(0.1, target_theta=1), 
        SimpleTest(0.01, target_theta=2), 
        SimpleTest(0.2, target_theta=3)]

# Create all the solvers
methods = [equivalence_class_solvers.ECED,
           equivalence_class_solvers.EC_bayes,
           equivalence_class_solvers.EC_US,
           equivalence_class_solvers.EC_IG,
           equivalence_class_solvers.EC_VoI]

solvers = [method(Theta, r, tests, theta_prior, true_theta = Theta[0]) for method in methods]
solver_names = [solver.get_name() for solver in solvers]
n_solvers = len(solvers)

for solver in solvers:
    test_num,result = solver.step()
    y_predicted,map_p_y = solver.predict_y()
    print '{0:<10}|{1:<8}|{2:<8}|{3:<8}|{4:0.3f}'.format(solver.get_name(), test_num, result, y_predicted, map_p_y)
