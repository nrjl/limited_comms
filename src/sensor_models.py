import numpy as np

class BinaryLogisticObs(object):
    def __init__(self, r=30.0, true_pos=0.95, true_neg=0.9, decay_rate=0.25):
        self.r = r
        self.true_pos = true_pos
        self.true_neg = true_neg
        self.decay_rate = decay_rate
        self._n_returns = 2

    def get_n_returns(self):
        return self._n_returns

    def likelihood(self, x, z, c):
        if z == True:
            return self.true_pos - (self.true_pos + self.true_neg - 1.0) / (
                1 + np.exp(-self.decay_rate * (np.linalg.norm(x[0:len(c)] - c) - self.r)))
        else:
            return 1.0 - self.likelihood(x, True, c)

    def generate_observations(self, x, c):
        # Generate observations at an array of locations
        obs=[]
        for xx in x:
            p_z = np.array([self.likelihood(xx, z, c) for z in range(self.get_n_returns())])
            zz = np.sum(np.random.uniform() > p_z.cumsum())
            obs.append((xx,zz))
        return obs

    def plot(self, axobs, n_range=3.0, nx=101):
        xx = np.atleast_2d(np.linspace(0, n_range*self.r, nx)).T
        yy = [self.likelihood(x, True, c=np.array([0.0])) for x in xx]
        axobs.plot(xx, yy)
        axobs.set_ylim(0, 1.0)
        axobs.set_xlim(0, xx[-1][0])
        axobs.set_ylabel(r'$P(z(r) = T)$')
        axobs.set_xlabel('Range, $r$')


class DiscreteStep(BinaryLogisticObs):
    def __init__(self, r=30.0, true_pos=0.95, true_neg=0.9):
        self.r = r
        self.true_pos = true_pos
        self.true_neg = true_neg
        self._n_returns = 2

    def likelihood(self, x, z, c):
        if z == True:
            return self.true_pos - (self.true_pos + self.true_neg - 1.0) * (np.linalg.norm(x[0:len(c)] - c) > self.r)
        else:
            return 1.0 - self.likelihood(x, True, c)



# # Discrete function (step)
# def step_obs(x,z,c,r=30.0, true_pos=0.95, true_neg=0.9):
#     if z==True:
#         return true_pos - (true_pos+true_neg-1.0)*(np.linalg.norm(x[0:len(c)]-c) > r)
#     else:
#         return 1.0 - step_obs(x,True,c,r,true_pos,true_neg)
#
#
#
# # Continuous function (logistic)
# def logistic_obs(x,z,c,r=30.0, true_pos=0.95, true_neg=0.9, decay_rate=0.25):
#     # High decay rate is sharp dropoff
#     if z==True:
#         return true_pos - (true_pos+true_neg-1.0)/(1+np.exp(-decay_rate*(np.linalg.norm(x[0:len(c)]-c)-r)))
#     else:
#         return 1.0-logistic_obs(x,True,c,r, true_pos, true_neg)
        
def arc_samples(c, r, t1, t2, n=6):
    ang = np.linspace(t1*np.pi/180, t2*np.pi/180,n)
    x = c[0]+r*np.cos(ang)
    y = c[1]+r*np.sin(ang)
    return np.array(zip(x,y))
    
def obs_subsampler(obsX,c,gridsize,nmax=None):
    c = np.array(c)
    obs = []
    ndim = len(c)
    # Lims is 2*ndims (first row low lims, second row high lims)
    lims = np.array([[-c[i],gridsize[i]-c[i]] for i in range(ndim)]).transpose()
    for x in obsX:
        if np.all(x > lims[0] and x < lims[1]):
            obs.append(x+c)
            if len(obs)>nmax:
                break
    return obs