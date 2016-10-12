import numpy as np
import itertools
import random

# Based on the lottery payoff learning problem from:
# Golovin, Krause, Roy. "Near-Optimal Bayesian Active Learning with Noisy 
# Observations." NIPS 2010

fancy_L = [-10, 0, 10]
pp = [.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.99]#0.0,

# Generate all possible lotteries (the dumb way...)
p_L = []
for p1 in pp:
    for p2 in pp:
        for p3 in pp:
            if p1+p2+p3 == 1.0:
                p_L.append((p1,p2,p3))
p_L = set(p_L)

# A particular lottery L is defined by a set of probabilities L=l_i

# Generate tests:
test_set = list(itertools.combinations(p_L,2))

# Hypotheses:
def EV(prob_L, *args):
    U = 0.0
    for li,pi in zip(fancy_L,prob_L):
        U += li*pi
    return U
    
def PT(prob_L, params):
    rho,lambd,alpha = params
    U = 0.0
    for li,pi in zip(fancy_L,prob_L):
        if li >= 0:
            f = li**rho
        else:
            f = -lambd*((-li)**rho)
        w = np.exp(-(np.log(1.0/pi))**alpha)
        U += f*w
    return U
    
def CRRA(prob_L, a):
    U = 0.0
    for li,pi in zip(fancy_L,prob_L):
        if a != 1.0:
            U += pi*(li**(1-a))/(1-a)
        else:
            U += pi*np.log(li)
    return U

theta_EV = []
theta_PT = [.9,2.2,.9]
theta_CRRA = 1.0

def binary_softmax(test,U,U_param):
    p0 = 1.0/(1.0 + np.exp(U(test[1],U_param)-U(test[0],U_param)))
    return [p0,1.0-p0]

def softmax_selector(test,U,U_param):
    p = binary_softmax(test,U,U_param)
    return random.uniform < p[0]
    
U_set = [EV,PT,CRRA]
U_param_set = [theta_EV,theta_PT,theta_CRRA]

# EffECXtive algorithm - boo acronym boo

# Class for each method:
class EC2Method(object):
    def __init__(self, hypotheses, tests, prior, test_cost, test_likelihood=binary_softmax):
        self.hypotheses = hypotheses
        self.tests = tests
        self.h_distribution = np.array(prior)
        self.test_cost = test_cost
        self.test_likelihood = test_likelihood
        self.reset()

    def reset(self):
        self.evidence = []
        self.available_tests = set(self.Tests)
    
    def utility(self, test):
        return random.uniform()    
                
    def select_test(self):
        max_util = None
        for test in self.available_tests:
            U = self.utility(test)
            if U > max_util:
                max_util = U/self.Cost(test)
                t_star = test
        self.available_tests.discard(t_star)
        return t_star
        
    def add_result(self, test, result):
        ph = self.test_likelihood(test)
        for 
        
    def get_MAP(self):
        # Break ties randomly
        m = np.amax(self.h_distribution)
        indices = np.nonzero(self.h_distribution == m)[0]
        return random.choice(indices)

        
            
            
        
