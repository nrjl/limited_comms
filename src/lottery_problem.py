import numpy as np
import random
import itertools
# import copy

# Based on the lottery payoff learning problem from:
# Chen, Hassani, Krause. "Near-optimal Bayesian Active Learning
# with Correlated and Noisy Tests." https://arxiv.org/abs/1605.07334

# A lottery is defined by a 2-tuple containing:
#        1 - set of probabilities [p_0, p_1, ..., p_n]
#        2 - associated payoffs [v_0, v_1, ..., v_n]

def lottery_builder(pp):
    # Generate all possible lottery probs from a set of probs (the dumb way...)
    p_L = set()
    for p1 in pp:
        for p2 in pp:
            for p3 in pp:
                if p1+p2+p3 == 1.0:
                    p_L.add((p1,p2,p3))                 
    return p_L

# If all have same payoff, generate vector of tests
def fixed_payoff_lottery_tests(prob, payoff, *args, **kwargs):
    lottery_pairs = list(itertools.combinations(prob,2))
    tests = [LotteryTest(Lottery(p0,payoff),Lottery(p1,payoff), *args, **kwargs) for p0,p1 in lottery_pairs]
    
    return tests        
    
# Broken recursive to build set of all lotteries
#def lottery_builder_loop(pp, loop_depth, lottery_set = set(), cval=[]):
#    if loop_depth == 0:
#        return lottery_set.add(tuple(cval))
#    else:
#        for p in pp:
#            cval2 = copy.copy(cval)
#            lottery_builder_loop(pp, loop_depth-1, lottery_set, cval2.append(p))

def binary_softmax(u0,u1,k=1.0):
    s0 = 1.0/(1.0 + np.exp(k*(u1-u0)))
    return [s0,1.0-s0]

# Lottery class:
class Lottery(object):
    def __init__(self, prob, payoff):
        self.prob = prob
        self.payoff = payoff
        self.isvalid()
        self._mean_var_skew()
        
    def isvalid(self):
        lp,lv = len(self.prob), len(self.payoff)
        if lp != lv:
            raise ValueError(('Vector size: length probs [{0}] doesn''t'+ 
                'match length payoff [{1}]').format(lp, lv))
        
        if abs(np.sum(self.prob) - 1.0) > 1e-5:
            raise ValueError('Probabilities don''t sum to 1.0')
            
    def _mean_var_skew(self):
        mean, var, skew = 0.0,0.0,0.0
        for pi,vi in zip(self.prob, self.payoff):
            mean += pi*vi
        for pi,vi in zip(self.prob, self.payoff):
            var += ((vi-mean)**2)*pi
            skew += ((vi-mean)**3)*pi
        self.mean, self.var, self.skew = mean, var, skew
        self.std = np.sqrt(var)
        self.sskew = skew/(self.std**3)

# Lottery test class (i.e comparison between two lotteries):
class LotteryTest(object):
    def __init__(self, L1, L2, cost=1.0, softmax_k=1.0):
        self.n_outcomes = 2
        self.L1 = L1
        self.L2 = L2
        self.cost = cost
        self.softmax_k = softmax_k
        
    def outcome_likelihood(self, theta):
        # A hypothesis theta is an object with a lottery_value(L) member fn
        U1 = theta.lottery_value(self.L1)
        U2 = theta.lottery_value(self.L2)
        return binary_softmax(U1,U2,k=self.softmax_k)

class EconomicUtility(object):
    def __init__(self, params):
        self.params = params
        self.extra_init()
        
    def extra_init(self):
        pass
        
    def _utility(self, lottery):
        return 1.0
        
    def lottery_value(self, lottery):
        if not isinstance(lottery, Lottery):
            raise ValueError('Object not Lottery class')
        return self._utility(lottery)
        
    @staticmethod
    def get_name():
        return 'NullMethod'
        
class ExpectedValue(EconomicUtility):
    def _utility(self, lottery):
        U = lottery.mean
        return U
        
    @staticmethod
    def get_name():
        return 'EV'

class MeanVarianceSkewness(EconomicUtility):
    def extra_init(self):
        try:
            self.w_mean, self.w_var,self.w_skew = self.params
        except ValueError:
            print 'Params input should contain three items; w_mean, w_var, w_skew'
            raise 
    
    def _utility(self, lottery):
        U = self.w_mean*lottery.mean - self.w_var*lottery.var + self.w_skew*lottery.skew
        return U
        
    @staticmethod
    def get_name():
        return 'MVS'

class StandardizedMeanVarianceSkewness(EconomicUtility):
    def extra_init(self):
        try:
            self.w_mean, self.w_std,self.w_sskew = self.params
        except ValueError:
            print 'Params input should contain three items; w_mean, w_std, w_sskew'
            raise 
    
    def _utility(self, lottery):
        U = self.w_mean*lottery.mean - self.w_std*lottery.std + self.w_sskew*lottery.sskew
        return U
        
    @staticmethod
    def get_name():
        return 'SMVS'              
                                          
class ProspectTheory(EconomicUtility):
    def extra_init(self):
        try:
            self.rho, self.lambd,self.alpha = self.params
        except ValueError:
            print 'Params input should contain three items; rho, lambda, alpha'
            raise 
    
    def _utility(self, lottery):
        U = 0.0
        for pi,vi in zip(lottery.prob, lottery.payoff):
            if vi >= 0:
                f = vi**self.rho
            else:
                f = -self.lambd*((-vi)**self.rho)
            w = np.exp(-(np.log(1.0/pi))**self.alpha)
            U += f*w
        return U

    @staticmethod
    def get_name():
        return 'PT'
        
class ConstantRelativeRiskAversion(EconomicUtility):
    def extra_init(self):
        #if not isinstance(self.params, float):
        #    raise ValueError('CRRA only has one parameter, a')
        try:
            self.a = self.params[0]
            self.w = self.params[1]
        except TypeError as ex:
            print 'CRRA requires two input parameters; [a,w]'
            raise ex
 
    def _utility(self, lottery):
        U = 0.0
        for pi,vi in zip(lottery.prob, lottery.payoff):
            if self.a != 1.0:
                U += pi*((self.w + vi)**(1-self.a))/(1-self.a)
            else:
                U += pi*np.log(self.w + vi)
        return U
        
    @staticmethod
    def get_name():
        return 'CRRA'


# class LotteryResult(object):
#     def __init__(self):
