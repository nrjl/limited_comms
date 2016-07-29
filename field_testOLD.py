import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

def calculate_entropy(fx):
    H = 0
    for val in np.nditer(fx):
        if val > 0:
            H = H - val*np.log(val)
    return H

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
    
    def update_plot(self,mat):
        self.hmat.set_data(mat.transpose())    
        self.hmat.set_clim([0, mat.max()])       

class TrueField:
    def __init__(self, gridsizex, gridsizey, centre, r, obs_mat, ax=None):      
        self.x = np.arange(gridsizex)
        self.y = np.arange(gridsizey)
        self.centre = np.array(centre)
        self.r = r
        self.obs_mat = obs_mat
        
        # Build field
        self.true_field = np.zeros([len(self.x),len(self.y)], dtype='int')
        for x in self.x:
            for y in self.y:
                if np.linalg.norm(np.array([x,y]) - self.centre) <= self.r:
                    self.true_field[x,y] = 1
        
        self.field = Field(self.true_field, ax)
        
    def generate_random_observations(self, n_obs):
        self.z_pos = np.hstack( (np.random.randint(len(self.x), size=(n_obs,1)), 
            np.random.randint(len(self.y), size=(n_obs,1)) ) )
        self.z = np.zeros(n_obs, dtype='bool')
        probs = np.random.uniform(size=(n_obs,1))
        for i, n_pos in enumerate(self.z_pos):
            if probs[i] < self.obs_mat[1,self.true_field[n_pos[0],n_pos[1]]]:
                self.z[i] = True

class ProbField:
    def __init__(self, gridsizex, gridsizey, r, obs_mat, ax=None):
        self.x = np.arange(gridsizex)
        self.y = np.arange(gridsizey)
        self.r = r
        self.obs_mat = obs_mat
        self.p_z = np.zeros(2)
        k = np.pi*r**2/(len(self.x)*len(self.y)) #+ 2*r*(len(self.x)+len(self.y)) + 2*r**2)
        self.p_z[0] = (1-k)*self.obs_mat[0,0] + k*self.obs_mat[0,1]
        self.p_z[1] = 1.0-self.p_z[0]
        self.field = Field(np.zeros((gridsizex,gridsizey)), ax, vmax=0.2)
        self.field_reset()
            
    def field_reset(self):
        r = self.r
        lx = len(self.x); ly = len(self.y);
        self.p_centre = np.zeros([lx,ly])
        self.p_centre[r:lx-r,r:ly-r] = 1.0/((lx-2*r)*(ly-2*r))*np.ones([lx-2*r,ly-2*r])
        self.field.update_plot(self.p_centre)

    def update_prob(self, x, z_loc, z):
        prior = self.p_centre[x[0],x[1]]
        in_range = 0
        if np.linalg.norm(x-z_loc) <= self.r:
            in_range = 1
        p_obs = self.obs_mat[z,in_range]
        #norma = self.p_z[z] #self.obs_mat[z,0]*prior + self.obs_mat[z,1]*(1-prior)
        self.p_centre[x[0],x[1]] = p_obs*prior/self.p_z[z]
        
    def update_field(self, z_loc, z):
        x_loc = np.array([0,0])
        for x_loc[0] in self.x:
            for x_loc[1] in self.y:
                self.update_prob(x_loc,z_loc,z)
        #self.p_centre = self.p_centre/self.p_centre.sum()
        self.field.update_plot(self.p_centre)
    
class ProbFieldSmooth:
    def __init__(self, gridsizex, gridsizey, obs_fun, ax=None, p_z = None):
        self.x = np.arange(gridsizex)
        self.y = np.arange(gridsizey)
        self.obs_fun = obs_fun
        self.field = Field(np.zeros((gridsizex,gridsizey)), ax, vmax=0.2)
        self.field_reset()
        self.p_z = p_z
        
    def field_reset(self):
        self.p_centre = 1.0/(len(self.x)*len(self.y))*np.ones([len(self.x),len(self.y)])
        self.field.update_plot(self.p_centre)

    def update_prob(self, x, z_loc, z):
        prior = self.p_centre[x[0],x[1]]
        r = np.linalg.norm(x-z_loc)
        p_obs = self.obs_fun[z](r)
        self.p_centre[x[0],x[1]] = p_obs*prior #/self.p_z[z]
        
    def update_field(self, z_loc, z):
        x_loc = np.array([0,0])
        for x_loc[0] in self.x:
            for x_loc[1] in self.y:
                self.update_prob(x_loc,z_loc,z)
        if self.p_z == None:
            self.p_centre = self.p_centre/self.p_centre.sum()
        else:
            self.p_centre = self.p_centre/self.p_z[z]
        self.field.update_plot(self.p_centre)

    
true_neg = 0.90     # P(obs_nothing|no_feature)
true_pos = 0.95     # P(obs_feature|feature)
target_radius = 10

obs_mat = np.array([[true_neg, 1-true_pos], [1-true_neg, true_pos]])

def obs_fundT(x):
    return true_pos - (true_pos+true_neg-1.0)*(x > target_radius)
def obs_fundF(x):
    return 1.0-obs_fundT(x)
obs_fund = [obs_fundF, obs_fundT]



def obs_funT(x):
    return true_pos - (true_pos+true_neg-1.0)/(1+np.exp(-1.0*(x-target_radius)))
def obs_funF(x):
    return 1.0-obs_funT(x)
obs_fun = [obs_funF, obs_funT]

gridsizex = 50
gridsizey = 80

target_centre = [30,40]

n_obs = 80

h_fig, h_ax = plt.subplots(1,2, sharex=True, sharey=True)
prob_field = ProbFieldSmooth(gridsizex, gridsizey, obs_fund, ax=h_ax[1])
#prob_field = ProbField(gridsizex, gridsizey, target_radius, obs_mat, ax=h_ax[1])
true_field = TrueField(gridsizex, gridsizey, target_centre, target_radius, obs_mat, ax=h_ax[0])

def init():
    true_field.generate_random_observations(n_obs)
    while len(h_ax[0].lines) > 0:
        h_ax[0].lines.pop(0)
    prob_field.field_reset()

#plt.show()
symbols = ('r^','go')

def animate(i, *args, **kwargs):
    z_pos = true_field.z_pos[i]
    prob_field.update_field(z_pos, true_field.z[i])
    h_ax[0].plot(z_pos[0],z_pos[1],symbols[true_field.z[i]])
    print true_field.z[i], prob_field.p_centre.sum(), calculate_entropy(prob_field.p_centre)
    return
    #wait = input("PRESS ENTER TO CONTINUE.")

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = False, repeat = True)
#ani.save('smooth_feature.mp4', writer = 'avconv', fps=5, bitrate=5000)

h_fig.show()