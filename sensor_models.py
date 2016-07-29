import numpy as np
# Discrete function (step)
def step_obs(x,z,c,r=30.0, true_pos=0.95, true_neg=0.9):
    if z==True:
        return true_pos - (true_pos+true_neg-1.0)*(np.linalg.norm(x-c) > r)
    else:
        return 1.0 - step_obs(x,True,c,r,true_pos,true_neg)
        
# Continuous function (logistic)
def logistic_obs(x,z,c,r=30.0, true_pos=0.95, true_neg=0.9):
    if z==True:
        return true_pos - (true_pos+true_neg-1.0)/(1+np.exp(-0.25*(np.linalg.norm(x-c)-r)))
    else:
        return 1.0-logistic_obs(x,True,c,r, true_pos=0.95, true_neg=0.9)
        
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