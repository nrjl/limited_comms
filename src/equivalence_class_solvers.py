import numpy as np
import copy
import random
import time

def discrete_entropy(p0):
    H = 0.0
    for p in (y for y in p0 if y > 0): # Get rid of log(0)
        H = H-p*np.log2(p)
    return H

def prior_flat_theta(r, Theta):
    pt = 1.0/len(Theta)
    return {t:pt for t in Theta}
    
def prior_flat_Y(r, Theta):
    py = 1.0/len(r)
    prior = {}
    for y in r:
        pt = py/len(y)
        for t in y:
            prior[t] = pt
    return prior

def select_from_prior(prior):
    return sum(random.random() > np.cumsum(prior))

def select_root_cause(r, theta, theta_prior):
    
    tp = [theta_prior[t] for t in theta]
    theta_true = theta[select_from_prior(tp)]

    # Work out what the correct action is from the true state
    y_true = -1
    for ny, y_set in enumerate(r):
        if theta_true in y_set:
            y_true = ny
    if y_true == -1:
        raise LookupError('Root cause {0} not found in r'.format(theta_true))

    return theta_true, y_true

class ExampleTest(object):
    # Example useless test to demonstrate minimum requirements
    def __init__(self, epsilon=0.05, cost=1.0):
        self.n_outcomes = 2
        self.epsilon = epsilon
        self.cost = cost
        
    def outcome_likelihood(self, theta):
        return [1.0-self.epsilon, self.epsilon]
    
class EC_solver(object):
    # Theta is an array of hypothesis objects
    # r is an array where each element is a set() of hypotheses, so union(r) = Theta
    # tests is an array of test objects, where each object must have member:
    #       - n_outcomes integer variable 
    #       - cost float variable that returns the cost of running the test
    #       - outcome_likelihood(theta) fn that returns a vector of outcome probabilities given a root cause
    def __init__(self, theta, r, tests, prior_theta, true_theta=[], verbose=False, test_repeats=False, test_outcomes=None):
        self.theta = theta
        self.r = r
        self.tests = tests
        self.n_tests = len(self.tests)
        self._available_tests = range(self.n_tests)
        self.prior_theta = prior_theta
        self.verbose = verbose
        self.test_repeats = False       # Allow tests to be performed multiple times
        self.test_outcomes = test_outcomes    # Prior specification of test outcomes (persistent noise)
        if verbose:
            print ('Constructing EC solver object with {0} hypotheses, {1} classes '+
            'and {2} tests.').format(len(self.theta),len(self.r),self.n_tests)
        
        # Build lambda matrix and test likelihoods
        if self.verbose:
            tnow = time.time()
            print 'Constructing test likelihood and lambda matrices...',
        self.lambda_full = {}
        self.test_likelihoods = {}
        for theta in self.theta:
            lambda_theta = [] # Tests are ordered!!
            like_theta = []
            for test in self.tests:
                pXe = np.array(test.outcome_likelihood(theta))
                maxpXe = max(pXe)
                lambda_theta.append(pXe/maxpXe) #[pxe/maxpXe for pxe in pXe])
                like_theta.append(pXe)
            self.lambda_full[theta] = lambda_theta
            self.test_likelihoods[theta] = like_theta
        if self.verbose:
            print 'done in {0}s'.format(time.time()-tnow)
        self._method_name = 'EC Default'
        self.init_extras()
        self.reset(true_theta)
        
    def reset(self, true_theta, test_outcomes=None):
        self.p_theta = copy.copy(self.prior_theta)
        self.p_Y = np.zeros(len(self.r))
        self.update_p_Y()
        self.evidence = []
        self.true_theta = true_theta
        if test_outcomes is not None:
            self.test_outcomes = test_outcomes
        self._available_tests = range(self.n_tests)
        self.reset_extras()

    def init_extras(self):
        return

    def reset_extras(self):
        return
    
    def calculate_p_Y(self, p_theta):
        p_Y = np.zeros(len(self.r))
        for ii,rY in enumerate(self.r):
            p_Yi = 0.0
            for theta in rY:
                p_Yi += p_theta[theta]
            p_Y[ii] = p_Yi
        return p_Y
                        
    def update_p_Y(self):
        self.p_Y = self.calculate_p_Y(self.p_theta)

    def select_test(self):
        return 0
    
    def run_test(self,test_num):
        if self.test_outcomes is None:
            pXe = self.tests[test_num].outcome_likelihood(self.true_theta)
            test_result = sum(random.random() > np.cumsum(pXe))
        else:
            test_result = self.test_outcomes[test_num]
        return test_result
        
    def add_result(self, test_num, result):
        # Add to evidence list
        self.evidence.append((test_num,result))
        
        # Update theta posterior
        p_accumulator = 0.0
        for theta in self.theta:
            p_xe = self.test_likelihoods[theta][test_num][result]
            self.p_theta[theta] = self.p_theta[theta]*p_xe
            p_accumulator += self.p_theta[theta]
        # Renormalize
        for theta in self.theta:
            self.p_theta[theta] = self.p_theta[theta]/p_accumulator
        
        # Update pY from new p_theta
        self.update_p_Y()
        self.add_result_extras(test_num,result)
        
    def add_result_extras(self,test_num,result):
        return
    
    def step(self):
        best_test = self.select_test()
        result = self.run_test(best_test)
        self.add_result(best_test, result)
        if not self.test_repeats:
            self._available_tests.remove(best_test)
        return best_test,result
                                
    def predict_y(self):
        # Break ties randomly
        m = np.max(self.p_Y)
        indices = np.nonzero(self.p_Y == m)[0]
        return random.choice(indices),m

    def get_name(self):
        return self._method_name
    

class ECED(EC_solver):
    def init_extras(self):
        self._method_name = 'ECED'

        # Construct edges:
        if self.verbose:
            tnow = time.time()
            print 'Constructing EC graph and offsets...',
        self.E = set()
        theta_set = set(self.theta)
        for rY in self.r:
            other_thetas = theta_set.difference(rY)
            for t in rY:
                for tprime in other_thetas:
                    if (tprime,t) not in self.E:
                        self.E.add((t,tprime))
                    
        # Build offsets (since they can be precalculated)
        self.offsets = []
        for test_num in self._available_tests:
            test = self.tests[test_num]
            offset_test = np.zeros(test.n_outcomes, dtype='float')
            for test_result in range(test.n_outcomes):
                alltheta = [self.lambda_full[theta][test_num][test_result] for theta in self.theta]
                offset_test[test_result] = 1.0 - (max(alltheta))**2
            self.offsets.append(offset_test)
        
        self.build_edge_lambda()
                    
        if self.verbose:
            print 'done in {0}s.'.format(time.time()-tnow)

    def reset_extras(self):
        self.w_E = {}
        for e in self.E:
            self.w_E[e] = self.p_theta[e[0]]*self.p_theta[e[1]]
        # self.update_delta_diff_mat()
    
    def build_edge_lambda(self):
        self.edge_lambda = {}
        for edge in self.E:
            test_lambda = []
            for test_num,test in enumerate(self.tests):
                lam_theta = self.lambda_full[edge[0]][test_num] 
                lam_thetap = self.lambda_full[edge[1]][test_num]
                offset = self.offsets[test_num]
                test_lambda.append(1.0-lam_theta*lam_thetap-offset)
            self.edge_lambda[edge] = test_lambda
               
    def select_test(self):
        delta_diff_mat = []
        for test_num,test in enumerate(self.tests):
            delta_diff_mat.append(np.zeros(test.n_outcomes, dtype='float'))
        
        for edge in self.E:            # Remember edge is theta pair
            for test_num in self._available_tests:
                e_test = self.w_E[edge]*self.edge_lambda[edge][test_num]
                delta_diff_mat[test_num] += e_test       
        
        max_util = None
        for test_num in self._available_tests:
            test = self.tests[test_num]
            # Now, for each possible outcome of each test
            U = 0.0
            for test_result in range(test.n_outcomes):
                p_xe = 0.0
                for theta in self.theta:
                    p_xe += self.p_theta[theta]*self.test_likelihoods[theta][test_num][test_result]
                # U += p_xe*(self.delta_BS(test_num,test_result) - self.delta_OFF(test_num,test_result))
                # U += p_xe*self.delta_diff(test_num,test_result)
                U += p_xe*delta_diff_mat[test_num][test_result]
            # U /= test.cost
            if U > max_util:
                max_util = U
                e_star = test_num
        return e_star
    
    def add_result_extras(self, test_num, result):
        self.update_edge_weights(test_num, result)
        
    def update_edge_weights(self, test_num, result):
        for e in self.E:
            px_t1 = self.test_likelihoods[e[0]][test_num][result]
            px_t2 = self.test_likelihoods[e[1]][test_num][result]
            self.w_E[e] = self.w_E[e]*px_t1*px_t2        

class EC_bayes(ECED):
    def init_extras(self):
        super(EC_bayes, self).init_extras()
        self._method_name = 'ECBayes'
    
    def build_edge_lambda(self):
        self.edge_lambda = {}
        for edge in self.E:
            test_lambda = []
            for test_num,test in enumerate(self.tests):
                px_t1 = self.test_likelihoods[edge[0]][test_num]
                px_t2 = self.test_likelihoods[edge[1]][test_num]
                test_lambda.append((1.0-px_t1*px_t2))
            self.edge_lambda[edge] = test_lambda
  

class EC_random(EC_solver):
    def init_extras(self):
        self._method_name = 'Random'

    def select_test(self):
        return random.choice(self._available_tests)


class EC_US(EC_solver):
    def init_extras(self):
        self._method_name = 'US'

    def select_test(self):
        # Maximize reduction of entropy over theta
        H_theta = discrete_entropy(self.p_theta.values())
        max_util = None
        p_theta_new = np.zeros(len(self.theta))
        for test_num in self._available_tests:
            test = self.tests[test_num]
            U = 0.0
            for test_result in range(test.n_outcomes):
                for ti,theta in enumerate(self.theta):
                    p_theta_new[ti] = self.p_theta[theta]*self.test_likelihoods[theta][test_num][test_result]
                p_xe = p_theta_new.sum()
                p_theta_new = p_theta_new/p_xe
                U += p_xe*(H_theta - discrete_entropy(p_theta_new))
            # U /= test.cost
            if U > max_util:
                max_util = U
                e_star = test_num
        return e_star
                        
class EC_IG(EC_solver):
    def init_extras(self):
        self._method_name = 'IG'

    def select_test(self):
        # Maximize reduction of entropy over Y
        H_Y = discrete_entropy(self.p_Y)
        max_util = None
        p_theta_new = {}
        for test_num in self._available_tests:
            test = self.tests[test_num]
            U = 0.0
            for test_result in range(test.n_outcomes):
                for theta in self.theta:
                    p_theta_new[theta] = self.p_theta[theta]*self.test_likelihoods[theta][test_num][test_result]
                p_xe = sum(p_theta_new.values())
                for t in p_theta_new:
                    p_theta_new[t] /= p_xe
                p_Y_new = self.calculate_p_Y(p_theta_new)
                U += p_xe*(H_Y - discrete_entropy(p_Y_new))
            # U /= test.cost
            if U > max_util:
                max_util = U
                e_star = test_num
        return e_star

class EC_JP(EC_solver):
    # Reduction in joint class probabilities
    # NOT FINISHED YET - DO NOT USE
    def init_extras(self):
        self._method_name = 'JP'

    def select_test(self):
        # Maximize reduction of entropy over Y
        H_Y = discrete_entropy(self.p_Y)
        max_util = None
        p_theta_new = {}
        for test_num,test in enumerate(self.tests):
            U = 0.0
            for test_result in range(test.n_outcomes):
                for theta in self.theta:
                    p_theta_new[theta] = self.p_theta[theta]*self.test_likelihoods[theta][test_num][test_result]
                p_xe = sum(p_theta_new.values())
                for t in p_theta_new:
                    p_theta_new[t] /= p_xe
                p_Y_new = self.calculate_p_Y(p_theta_new)
                U += p_xe*(H_Y - discrete_entropy(p_Y_new))
            # U /= test.cost
            if U > max_util:
                max_util = U
                e_star = test_num
        return e_star   
         
class EC_VoI(EC_solver):
    def init_extras(self):
        self._method_name = 'VoI'
        
    def select_test(self):
        # Minimize prediction error in Y
        max_util = None
        current_err_Y = 1.0 - max(self.p_Y)
        p_theta_new = {}
        for test_num in self._available_tests:
            test = self.tests[test_num]
            U = 0.0
            for test_result in range(test.n_outcomes):
                for theta in self.theta:
                    p_theta_new[theta] = self.p_theta[theta]*self.test_likelihoods[theta][test_num][test_result]
                p_xe = sum(p_theta_new.values())
                for t in p_theta_new:
                    p_theta_new[t] /= p_xe
                p_Y_new = self.calculate_p_Y(p_theta_new)
                U += p_xe*(current_err_Y - (1.0-max(p_Y_new)))
            # U /= test.cost
            if U > max_util:
                max_util = U
                e_star = test_num
        return e_star
                