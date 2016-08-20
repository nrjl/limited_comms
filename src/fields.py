import numpy as np
import matplotlib.pyplot as plt

class Field:
    def __init__(self, mat, ax=None, *args, **kwargs):
        if ax==None:
            (_, self.h_ax) = plt.subplot(111)
        else:
            self.h_ax = ax
        self.hmat = self.h_ax.imshow(mat.transpose(), origin='lower',vmin=0, *args, **kwargs)
        self.h_ax.set_aspect('equal')
        self.h_ax.tick_params(labelbottom='on',labeltop='off')
        self.h_ax.set_xlabel(r'x')
        self.h_ax.set_ylabel(r'y')
        self.h_ax.set_xlim(0,mat.shape[0])
        self.h_ax.set_ylim(0,mat.shape[1])
    
    def update_plot(self,mat,fmax=None):
        self.hmat.set_data(mat.transpose())
        if fmax == None:
            fmax = mat.max()
        self.hmat.set_clim([0, fmax])       

class TrueField:
    def __init__(self, gridsizex, gridsizey, obs_fun, obs_fun_kwargs={}, ax=None):      
        self.x = np.arange(gridsizex)
        self.y = np.arange(gridsizey)
        self.obs_fun = obs_fun
        self.obs_fun_kwargs = obs_fun_kwargs
        
        # Build field
        self.true_field = np.zeros([len(self.x),len(self.y)], dtype='float')
        for x in self.x:
            for y in self.y:
                self.true_field[x,y] = self.obs_fun[1](np.array([x,y]), **self.obs_fun_kwargs)        
        self.field = Field(self.true_field, ax,vmax=1.0)
        
    def generate_random_observations(self, n_obs):
        self.z_pos = np.hstack( (np.random.randint(len(self.x), size=(n_obs,1)), 
            np.random.randint(len(self.y), size=(n_obs,1)) ) )
        self.z = np.zeros(n_obs, dtype='bool')
        probs = np.random.uniform(size=(n_obs,1))
        for i, n_pos in enumerate(self.z_pos):
            if probs[i] < self.true_field[n_pos[0],n_pos[1]]:
                self.z[i] = True

class ProbField:
    def __init__(self, gridsizex, gridsizey, obs_fun, obs_fun_kwargs = {}, ax=None, p_z = None):
        self.x = np.arange(gridsizex)
        self.y = np.arange(gridsizey)
        self.obs_fun = obs_fun
        self.obs_fun_kwargs = obs_fun_kwargs
        self.field = Field(np.zeros((gridsizex,gridsizey)), ax, vmax=0.2)
        self.field_reset()
        self.p_z = p_z
        
    def field_reset(self):
        self.p_centre = 1.0/(len(self.x)*len(self.y))*np.ones([len(self.x),len(self.y)])
        self.field.update_plot(self.p_centre)

    def update_prob(self, x, z_loc, z):
        prior = self.p_centre[x[0],x[1]]
        p_obs = self.obs_fun[z](z_loc, x, **self.obs_fun_kwargs)
        self.p_centre[x[0],x[1]] = p_obs*prior
        
    def update_field(self, z_loc, z, fmax=None):
        x_loc = np.array([0,0])
        for x_loc[0] in self.x:
            for x_loc[1] in self.y:
                self.update_prob(x_loc,z_loc,z)
        if self.p_z == None:
            self.p_centre = self.p_centre/self.p_centre.sum()
        else:
            self.p_centre = self.p_centre/self.p_z[z]
        self.field.update_plot(self.p_centre, fmax=fmax)
