import numpy as np
import random
import lottery_problem
import equivalence_class_solvers
import plot_lottery
import time
import os
import pickle
import matplotlib.pyplot as plt

n_sims =  400                 # Number of simulated tests to run
max_tests =  51               # Max tests in each simulation
termination_threshold = 0.00  # Max probability of error threshold
seed_start = 200              # Random seed start

softmax_k = 10.0              # Logistic function steepness (higher is less noise)

verbose = True
write_files = True
persistent_noise = True       # Test results are fixed with persistent noise, and tests are not repeated

# Based on the lottery payoff learning problem from:
# Chen, Hassani, Krause. "Near-optimal Bayesian Active Learning
# with Correlated and Noisy Tests." https://arxiv.org/abs/1605.07334

# NOTE THAT TESTS CANNOT BE REPEATED!! (It is assumed that the outcome will not change, so-called persistent noise)

# Time stamp string
nowstr = time.strftime("%Y_%m_%d-%H_%M")
outdir = '../data/lottery/'+nowstr+'/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Construct test lotteries
lottery_payoff = (-10, 0, 10)
pp = [.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99]#0.0,
p_lottery_set = lottery_problem.lottery_builder(pp)
tests = lottery_problem.fixed_payoff_lottery_tests(p_lottery_set, lottery_payoff, softmax_k=softmax_k)

# Create hypotheses (parameterised economic models)
crra_theta = [.2, .4, .6, .8, 1.0]
crra_wealth = [100]
crra_all = [(t, w) for t in crra_theta for w in crra_wealth]

pt_rho = [0.5, 0.74, 0.86, 1.1] # [0.5, 0.62, 0.74, 0.86, 0.98, 1.1]
pt_lambda = [1, 2, 3.0] # [1, 1.5, 2, 2.5, 3.0]
pt_alpha = [0.4, 0.6, 0.8, 1.0] # [0.4, 0.52, 0.64, 0.76, 0.88, 1.0]
pt_all = [(r,l,a) for r in pt_rho for l in pt_lambda for a in pt_alpha]

mvs_w_mean = [0.6, 0.7, 0.8, 0.9, 1.0]
mvs_w_var = [0.0005, 0.0025, 0.0045, 0.0065, 0.085,0.0105]
mvs_w_skew = [0, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
mvs_all = [(m,v,s) for m in mvs_w_mean for v in mvs_w_var for s in mvs_w_skew]

smvs_w_mean = [0.6, 0.8, 1.0] # [0.6, 0.7, 0.8, 0.9, 1.0]
smvs_w_std = [0.05, 0.25, 0.45] # [0.05, 0.15, 0.25, 0.35, 0.45]
smvs_w_sskew = [0, 0.2, 0.35, 0.5] # [0, 0.1, 0.2, 0.3, 0.4, 0.5]
smvs_all = [(m,v,s) for m in smvs_w_mean for v in smvs_w_std for s in smvs_w_sskew]

method_pairs = [(lottery_problem.ExpectedValue, [0]), 
            (lottery_problem.ConstantRelativeRiskAversion, crra_all), 
            (lottery_problem.ProspectTheory, pt_all),
            #(lottery_problem.MeanVarianceSkewness, mvs_all),
            (lottery_problem.StandardizedMeanVarianceSkewness, smvs_all)]

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

solvers = [method(Theta, r, tests, theta_prior, Theta[0], verbose=verbose, test_repeats=(not persistent_noise)) for method in methods]
solver_names = [solver.get_name() for solver in solvers]  # 'EC Bayes', , 'IG'
n_solvers = len(solvers)

# Binary matrix of whether the action prediction was correct at each step
correct_predictions = np.zeros((n_sims, max_tests, n_solvers), dtype='bool')

# Counts of the number of successful choices of action at each step of simulations
n_correct = np.zeros((max_tests, n_solvers))

# Number of steps to termination
steps_to_term = np.ones((n_sims, n_solvers), dtype='int')*max_tests
correct_at_term = np.zeros((n_sims, n_solvers), dtype='bool')

# Prediction confidence (mean MAP p(y))
mean_map_p = np.zeros((max_tests, n_solvers), dtype='float')
map_p = -1*np.ones((n_sims, max_tests, n_solvers), dtype='float')

# This is a sanity check to check how many times each test is being selected by each method
test_picks = np.zeros((len(tests), n_solvers), dtype='int')
test_results=None

for ii in range(n_sims):
    random.seed(seed_start+ii)
    
    # For each simulation, select a random true root cause
    theta_true, y_true = equivalence_class_solvers.select_root_cause(r, Theta, theta_prior)
    print "Run {0}/{1} - True theta: {2}, True y: {3}".format(ii+1, n_sims, theta_true, Y[y_true])
    if verbose:
        print '{0:<10}|{1:<8}|{2:<8}|{3:<8}|{4:<8}|{5}'.format('Method', 
            'Test', 'Result', 'Time (s)', 'Y_max', 'p(Y)')

    # If using persistent noise, pre-calculate all test outcomes:
    if persistent_noise:
        pXe_all = [test.outcome_likelihood(theta_true) for test in tests]
        test_results = [sum(random.random() > np.cumsum(pXe)) for pXe in pXe_all]

    # Reset all the solvers with the true state
    for solver in solvers:
        solver.reset(theta_true, test_outcomes=test_results)

    # Counter for number of tests and which tests are still active
    active_tests = np.ones(n_solvers, dtype='bool')
    jj = 0

    # Run solvers
    while any(active_tests):
        if verbose:
            print '{0}---------------'.format(jj)
        
        for ns, solver in enumerate(solvers):
            if active_tests[ns]:
                tnow = time.time()
                test_num,result = solver.step()
                y_predicted,map_p_y = solver.predict_y()
                if verbose:
                    print '{0:<10}|{1:<8}|{2:<8}|{3:8.2f}|{4:<8}|{5:0.3f}|{6:<8}'.format(
                        solver.get_name(), test_num, result, time.time()-tnow, Y[y_predicted], map_p_y, len(solver._available_tests))

                correct = (y_predicted == y_true)
                correct_predictions[ii,jj,ns] = correct
                if correct:
                    n_correct[jj, ns] += 1
                test_picks[test_num, ns] += 1
                map_p[ii, jj, ns] += map_p_y
                mean_map_p[jj, ns] += (map_p_y-mean_map_p[jj, ns])/(ii+1)
                if (jj+1 >= max_tests) or (1.0 - map_p_y < termination_threshold):
                    active_tests[ns] = False
                    steps_to_term[ii, ns] = jj+1
                    correct_at_term[ii, ns] = correct
        jj += 1
    for ns, solver in enumerate(solvers):
        print

if write_files:
    with open(outdir+'results.pkl', 'wb') as fh:
        pickle.dump(correct_predictions, fh)
        pickle.dump(map_p, fh)
        pickle.dump(n_correct, fh)
        pickle.dump(mean_map_p, fh)
        pickle.dump(steps_to_term, fh)

print "Mean tests at termination: {0}".format(np.mean(steps_to_term, axis=0))
print "Correct at termination: {0}".format(['{:.2f}'.format(i) for i in np.sum(correct_at_term, axis=0)*100.0/n_sims])

hf, ha = plot_lottery.plot_action_predictions(solver_names, n_correct, steps_to_term, max_tests)
hpf, hpa = plot_lottery.plot_error_probabilities(solver_names, mean_map_p)
hcf, hca = plot_lottery.plot_correct_at_term(solver_names, correct_at_term, n_sims)

plt.show()
