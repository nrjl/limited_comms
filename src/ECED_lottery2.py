import numpy as np
import random
import lottery_problem
import equivalence_class_solvers
import matplotlib.pyplot as plt

n_sims =  5                  # Number of simulated tests to run
max_tests = 100               # Max tests in each simulation
termination_threshold = 0.05  # Max probability of error threshold
    
# Based on the lottery payoff learning problem from:
# Chen, Hassani, Krause. "Near-optimal Bayesian Active Learning
# with Correlated and Noisy Tests." https://arxiv.org/abs/1605.07334

# Construct test lotteries
lottery_payoff = [-10, 0, 10]
pp = [.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99]#0.0,
p_lottery_set = lottery_problem.lottery_builder(pp)
tests = lottery_problem.fixed_payoff_lottery_tests(p_lottery_set, lottery_payoff)

# Create hypotheses (parameterised economic models)
crra_theta = [.2, .4, .6, .8, 1.0]
crra_wealth = 100
crra_all = [(t, crra_wealth) for t in crra_theta]

pt_rho = [0.5, 0.62, 0.74, 0.86, 0.98, 1.1]
pt_lambda = [1, 1.5, 2, 2.5, 3.0]
pt_alpha = [0.4, 0.52, 0.64, 0.76, 0.88, 1.0]
pt_all = [(r,l,a) for r in pt_rho for l in pt_lambda for a in pt_alpha]

method_pairs = [(lottery_problem.ExpectedValue, [0]), 
            (lottery_problem.ConstantRelativeRiskAversion, crra_all), 
            (lottery_problem.ProspectTheory, pt_all)]

# Create Theta (ordered array of hypotheses) and r (array of theta sets grouped by Y):
Y = []
r = []
Theta = []
for method,params in method_pairs:
    Y.append(method.get_name())
    method_theta = [method(p) for p in params]
    r.append(set(method_theta))
    Theta.extend(method_theta)

# Theta prior (dictionary: uniform across classes, uniform in class)
theta_prior = equivalence_class_solvers.prior_flat_Y(r,Theta)

# Create all the solvers
methods = [equivalence_class_solvers.ECED,
           equivalence_class_solvers.EC_bayes,
           equivalence_class_solvers.EC_random,
           equivalence_class_solvers.EC_US,
           equivalence_class_solvers.EC_IG,
           equivalence_class_solvers.EC_VoI]

solvers = [method(Theta, r, tests, theta_prior, Theta[0]) for method in methods]
solver_names = [solver.get_name() for solver in solvers]  # 'EC Bayes', , 'IG'
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

random.seed(0)
random.jumpahead(500)

for ii in range(n_sims):
    # For each simulation, select a random true root cause
    theta_true, y_true = equivalence_class_solvers.select_root_cause(r, Theta, theta_prior)
    print "Run {0}/{1} - True theta: {2}, True y: {3}".format(ii+1, n_sims, theta_true, Y[y_true])

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

#print test_picks
#hf, ha = plt.subplots()
#ha.boxplot(steps_to_term, labels=solver_names)
#ha.set_xlabel('Method')
#ha.set_ylabel('Observations to termination')
# ha.legend(loc=0)

hf, ha = plt.subplots()
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


print "Mean tests at termination: {0}".format(np.mean(steps_to_term, axis=0))
print "Correct at termination: {0}".format(['{:.2f}'.format(i) for i in np.sum(correct_at_term, axis=0)*100.0/n_sims])
# hpf,hpa = plt.subplots()
# hpa.bar(np.arange(n_solvers), np.sum(correct_at_term, axis=0)*100.0/n_sims)
# hpa.set_xlabel('Method')
# hpa.set_ylabel(r'Correct prediction at termination $\%$')
# hpa.legend(loc=0)

plt.show()
