import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
import belief_state
import copy
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
np.random.seed(2)

# Number of observations for simulation
n_obs = 100

# Truth data
field_size = (100,100)
target_centre = np.array([62.0,46.0])
target_radius = 15.0

# Observation model
obs_model = sensor_models.logistic_obs
obs_kwargs = {'r':target_radius,'true_pos':0.95,'true_neg':0.90}

# Start state (location and heading rad)
start_pose = np.array([18.0, 23.0, np.pi/2])

# Prior sampler
mcsamples = 1000

# Location sampler
rel_pos = np.array([[-2.0, 0.0],[-2.0,2.2],[0.0,4.0],[2.0,2.2],[2.0, 0.0]])
rel_ang = np.array([[np.pi],[np.pi/2],[0.0],[-np.pi/2],[-np.pi]])
def motion_function(cpose):
    cs = np.cos(cpose[2])
    ss = np.sin(cpose[2])
    pose_out = np.dot(rel_pos, np.array([[cs,-ss],[ss, -cs]]))
    pose_out = np.hstack((cpose[0:2]+pose_out, cpose[2]+rel_ang))
    return pose_out

# Other constants
obs_symbols = ['r^','go']

# Setup vehicle belief
vehicle_belief = belief_state.belief(field_size, obs_model, obs_kwargs)
vehicle_belief.set_motion_function(motion_function)

h_fig, h_ax = plt.subplots() #1,2, sharex=True, sharey=True)
h_cpos, = h_ax.plot([],[],'ro',fillstyle='full')
h_pc = h_ax.imshow(np.zeros(field_size), origin='lower',vmin=0)
h_ax.set_xlim(-.5, field_size[0]-0.5)
h_ax.set_ylim(-.5, field_size[1]-0.5)

def init():
    #h_ax[0].cla(); h_ax[1].cla()
    vehicle_belief.set_current_pose(copy.copy(start_pose))
    vehicle_belief.reset_observations()
    
    xs = vehicle_belief.uniform_prior_sampler(mcsamples)
    vehicle_belief.assign_prior_sample_set(xs)
   
    obs = vehicle_belief.generate_observations([start_pose[0:2]],c=target_centre)
    vehicle_belief.add_observations(obs)
    pc = vehicle_belief.centre_probability_map()
    h_pc.set_data(pc.transpose())
    h_pc.set_clim([0, pc.max()])       
    
    h_ax.plot(target_centre[0],target_centre[1],'wx',mew=2,ms=10)
    h_ax.plot(start_pose[0],start_pose[1],'g^',fillstyle='full')
    for xx,zz in obs:
        h_ax.plot(xx[0],xx[1],obs_symbols[zz])
    h_cpos.set_data(start_pose[0],start_pose[1])
    return h_cpos,

def animate(i):
    fposes,Vx = vehicle_belief.select_observation(target_centre)
    Vx = Vx-Vx.min()
    h_ax.plot(fposes[:,0],fposes[:,1],'k.') #,color=plt.cm.jet(Vx/Vx.max()) 
    obs = vehicle_belief.get_observations()[-1]
    xx,zz = obs
    h_ax.plot(xx[0],xx[1],obs_symbols[zz])
    
    #if i % 5 == 0:
    pc = vehicle_belief.centre_probability_map()
    h_pc.set_data(pc.transpose())
    h_pc.set_clim([0, pc.max()])    
    
    cpos = vehicle_belief.get_current_pose()
    h_cpos.set_data(cpos[0],cpos[1])   
    print "i = {0}/{1}".format(i,n_obs)
    return h_cpos,

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = False, repeat = False)
#ani.save('vid/single_tracking.mp4', writer = 'avconv', fps=5, bitrate=5000, extra_args=['-vcodec', 'libx264'])
h_fig.show()


## SCRAP
#import mcmc
## MCMC sampler to sample from p_z_given_c
#obs_lims = [[-field_size[0],-field_size[1]],[field_size[0],field_size[1]]]
#mcmc_obsF_obj = mcmc.MCMCSampler(obs_model, obs_lims, func_kwargs=dict(obs_kwargs, z=False, c=(0,0)))
#mcmc_obsFX,mcmc_obsFp = mcmc_obsF_obj.sample_chain(mcmc_n_samples, mcmc_burnin)
#mcmc_obsT_obj = mcmc.MCMCSampler(obs_model, obs_lims, func_kwargs=dict(obs_kwargs, z=True, c=(0,0)))
#mcmc_obsTX,mcmc_obsTp = mcmc_obsT_obj.sample_chain(mcmc_n_samples, mcmc_burnin)
#np.random.shuffle(mcmc_obsFX)
#np.random.shuffle(mcmc_obsTX)
#
#fig2,ax2 = plt.subplots()
#ax2.plot(mcmc_obsFX[::100,0],mcmc_obsFX[::100,1],obs_symbols[0])
#ax2.plot(mcmc_obsTX[::100,0],mcmc_obsTX[::100,1],obs_symbols[1])
