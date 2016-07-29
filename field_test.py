import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fields import ProbField,TrueField
from mcmc import MCMCSampler

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

def entropy(x):
    if x > 0:
        return -x*np.log(x)
    return 0
    
def array_entropy(fx):
    H = 0
    for val in np.nditer(fx):
        H += entropy(val)
    return H


true_neg = 0.90     # P(obs_nothing|no_feature)
true_pos = 0.95     # P(obs_feature|feature)
target_centre = [60,80]
target_radius = 30
gridsizex = 100
gridsizey = 150
n_obs = 10
h_fig, h_ax = plt.subplots(1,2, sharex=True, sharey=True)

# Discrete function (step)
def step_obsT(x,c,r):
    return true_pos - (true_pos+true_neg-1.0)*(np.linalg.norm(x-c) > r)
step_obs = [lambda x,c,r : 1.0-step_obsT(x,c,r), step_obsT]
p_zstep = np.zeros(2)
k = np.pi*target_radius**2/(gridsizex*gridsizey) #+ 2*r*(len(self.x)+len(self.y)) + 2*r**2)
p_zstep[0] = (1-k)*true_neg + k*(1-true_pos)
p_zstep[1] = 1.0-p_zstep[0]

# Continuous function (logistic)
def logistic_obsT(x,c,r):
    return true_pos - (true_pos+true_neg-1.0)/(1+np.exp(-0.25*(np.linalg.norm(x-c)-r)))
logistic_obs = [lambda x,c,r : 1.0-logistic_obsT(x,c,r), logistic_obsT]
p_zlog = np.array([0.83047,0.16953])

obs_fun = logistic_obs  # step_obs
p_z = p_zlog            # p_zstep

prob_field = ProbField(gridsizex, gridsizey, obs_fun, obs_fun_kwargs={'r': target_radius}, ax=h_ax[1], p_z=p_z)
true_field = TrueField(gridsizex, gridsizey, obs_fun, obs_fun_kwargs={'c': target_centre, 'r': target_radius}, ax=h_ax[0])

h_zf, h_za = plt.subplots(1, 1,figsize=(6,4))
rr = np.linspace(0, 2.0*target_radius, 100)
zz = [obs_fun[1](r,0,target_radius) for r in rr]
h_za.axhline(true_pos, c='cornflowerblue', linestyle='dashed', label=r'$\alpha$')
h_za.axhline(1-true_neg, c='darkgreen', linestyle='dashed', label=r'$1-\beta$')
h_za.plot(rr, zz, linewidth=2, label='Likelihood')
h_za.set_xlabel('Range, $r =||y - c||$')
h_za.set_ylabel('$P(z(y) = T)$')
h_za.set_ylim(0, 1.0)
hll = h_za.legend()
hll.get_frame().set_color('white')
h_zf.tight_layout()

def likelihood_product(f_like, arg_like, f_prior, arg_prior, c, y, z):
    assert y.shape[0] == len(z), "Observations number mismatch (y.shape[0] ~= len(z))"
    lp = f_prior(c, **arg_prior)
    for i,yi in enumerate(y):
        lp = lp*f_like[z[i]](yi,c,**arg_like)
    return lp
    
def prior_fun(x):
    return 1.0/(gridsizex*gridsizey)

#mc_f = lambda x: likelihood_product(logistic_obs, {'r': target_radius}, prior_fun, {}, x, 
#mcmc_field = MCMCSampler(mc_f, np.array([[0, 0], [gridsizex, gridsizey]]) )
n_samples = 5000
burn_in = 1000
h_mcmc, = h_ax[1].plot(gridsizex*np.random.rand(n_samples), gridsizey*np.random.rand(n_samples), 'k.')

def init():
    true_field.generate_random_observations(n_obs)
    while len(h_ax[0].lines) > 0:
        h_ax[0].lines.pop(0)
    prob_field.field_reset()

#plt.show()
symbols = ('r^','go')

def animate(i, *args, **kwargs):
    z_pos = true_field.z_pos[i]
    mc_f = lambda x: likelihood_product(logistic_obs, {'r': target_radius}, prior_fun, {}, x, true_field.z_pos[0:i+1,:], true_field.z[0:i+1])
    mcmc_field = MCMCSampler(mc_f, np.array([[0, 0], [gridsizex, gridsizey]]) )
    X,fX = mcmc_field.sample_chain(n_samples, burn_in)
    h_mcmc.set_data(X[:,0],X[:,1])
    
    prob_field.update_field(z_pos, true_field.z[i])
    h_ax[0].plot(z_pos[0],z_pos[1],symbols[true_field.z[i]])
    print true_field.z[i], prob_field.p_centre.sum(), array_entropy(prob_field.p_centre)
    return
    #wait = input("PRESS ENTER TO CONTINUE.")

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = False, repeat = False)
#ani.save('smooth.mp4', writer = 'avconv', fps=5, bitrate=5000)
h_fig.show()

