import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

class ProbField:
    def __init__(true_pos, true_new, r, gridsizex, gridsizey=None):
        self.x = np.arange(gridsizex)
        self.y = np.arange(gridsizey)
        self.r = r
        if gridsizey == None:
            gridsizey = gridsizex
        self.obs_mat = np.array([[true_pos, 1-true_pos], [1-true_neg, true_neg]])
        self.newgrid = np.meshgrid(x,y)
        self.p_centre = 1.0/(gridsizex*gridsizey)*np.ones([gridsizex,gridsizey])
        self.fh, self.ah = plt.subplot(111)
        self.ah.set_aspect('equal')
        self.ah.tick_params(labelbottom='on',labeltop='off')
        self.ah.set_xlabel(r'x')
        self.ah.set_ylabel(r'y')
        self.ah.autoscale(tight=True)

    def update_prob(x,z_loc,z,prior):
        in_range = 0
        if np.linalg.norm(x,y) <= self.r:
            in_range = 1
        p_obs = self.obs_mat[z,in_range]
        norma = self.obs_mat[z,0]*prior + self.obs_mat[z,1]*(1-prior)
        p_c = p_obs*prior/norma
        
    def update_field(z_loc, z):
        z_loc = np.array([0,0])
        for ix, z_loc[0] in enumerate(x):
            for iy, z_loc[1] in enumerate(y):
                p_centre[ix,iy] = update_prob([lx,ly],z_loc,z,self.p_centre[ix,iy])
    
    def plot_field(*args, **kwargs):
        return  [self.ah.imshow(self.p_centre.transpose(), origin='lower',*args,**kwargs)]
        


true_pos = 0.99     # P(obs_feature|feature)
true_neg = 0.95     # P(obs_nothing|no_feature)

# Simple test
p_e = 0.5
new_p = true_pos*p_e/(true_pos*p_e + (1-true_neg)*(1-p_e))
