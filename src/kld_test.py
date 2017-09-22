import numpy as np
import time
from sensor_models import  BinarySensor
import itertools


def kld(P, Q):
    return (P * np.log(P / Q)).sum()


class BinaryInferrer(object):

    def __init__(self, sensor, prior=None):
        self.sensor = sensor
        self.states = np.arange(self.sensor.get_n_states())
        if prior is None:
            prior = np.ones(self.sensor.get_n_states())*1.0/self.sensor.get_n_states()
        self.prior = prior
        self.pX = self.prior.copy()

    def add_observation(self, z):
        pzgX, pzandX, pz = self.observation_likelihood(z)
        self.pX = pzandX/pz

    def add_observations(self, Z):
        pZgX, pZandX, pZ = self.joint_observation_likelihood(Z)
        self.pX = pZandX / pZ

    def observation_likelihood(self, z):
        pzgX = self.sensor.likelihood([], z, self.states)
        pzandX = pzgX * self.pX
        pz = pzandX.sum()
        return pzgX, pzandX, pz

    def joint_observation_likelihood(self, Z):
        pZgX = 1.0
        for z in Z:
            pZgX *= self.sensor.likelihood([], z, self.states)
        pZandX = pZgX*self.pX
        pZ = pZandX.sum()
        return pZgX, pZandX, pZ

    def E_Dkl(self, depth, current_depth = 0, pJgX = 1.0):
        if current_depth >= depth:
            pJandX = pJgX*self.pX
            return (pJandX*(np.log(pJgX) - np.log(pJandX.sum()))).sum()

        E_d = 0.0
        for z in range(self.sensor.get_n_returns()):
            n_pJgX = pJgX*self.sensor.likelihood([], z, self.states)
            E_d += self.E_Dkl(depth, current_depth+1, n_pJgX)
        return E_d

    def E_Dkl2(self, depth):
        E_d = 0.0
        for Z in itertools.product(range(self.sensor.get_n_returns()), repeat=depth):
            pZgX, pZandX, pZ = self.joint_observation_likelihood(Z)
            E_d += pZ*kld(pZandX/pZ, self.pX)
        return E_d

test_sensor = BinarySensor([0.8, 0.3, 0.3])
bb = BinaryInferrer(test_sensor)
obs_list = [0, 1, 0, 0, 0]

for obs in obs_list:
    bb.add_observation(obs)
    print "P(X|z={z}) = {X}".format(z=obs, X=bb.pX)
    print "D_KL = {0:0.6f}".format(kld(bb.pX, bb.prior))
# print "D_{KL} = {0:0.4f}".format(1.0/bb.pI*(bb.pIgX*))

depth = 10
t_start = time.time()
E_d = bb.E_Dkl(depth)
t_total = time.time() - t_start
print "Recursive - E[D_kl] = {0}, t = {1}s".format(E_d, t_total)

t_start = time.time()
E_d2 = bb.E_Dkl2(depth)
t_total2 = time.time() - t_start
print "Iterator - E[D_kl] = {0}, t = {1}s".format(E_d2, t_total2)

# SCRAP
# bb1 = BinaryInferrer(test_sensor)
# bb1.add_observation(obs_list[0])
# DD = kld(bb.pX, bb1.pX)
# n_iter = 10000
# t_start = time.time()
# for i in range(n_iter):
#     pJgX = test_sensor.likelihood(obs_list[1], bb.states)
#     for obs in obs_list[2:]:
#         pJgX *= test_sensor.likelihood(obs, bb.states)
#     pIJandX = (pJgX*bb1.pIandX)
#     pIJ = pIJandX.sum()
#     D3 = 1.0/pIJ*( pIJandX*(np.log(pJgX) + np.log(bb1.pI/pIJ))).sum() # 1.0/pIJandX.sum()*( pIJandX*np.log(pJgX*bb1.pI/pIJ)).sum() #
# ttime = time.time()-t_start
# print "Total: {0}s, Avg:{1}s, Error:{2}".format(ttime, ttime/n_iter, D3 - DD)
#
# t_start = time.time()
# for i in range(n_iter):
#     pJgX = test_sensor.likelihood(obs_list[1], bb.states)
#     for obs in obs_list[2:]:
#         pJgX *= test_sensor.likelihood(obs, bb.states)
#     pIJandX = (pJgX * bb1.pX)
#     D4 = kld(pIJandX / pIJandX.sum(), bb1.pX)
# ttime = time.time()-t_start
# print "Total: {0}s, Avg:{1}s, Error:{2}".format(ttime, ttime/n_iter, D4 - DD)



# D2 = float(1.0/bb.pI*(bb.pIandX*np.log(bb.pIgX/bb.pI)).sum())