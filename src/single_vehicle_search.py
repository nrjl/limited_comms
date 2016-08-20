import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
from motion_models import yaw_rate_motion
import belief_state
import copy
plt.style.use('ggplot')
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
np.random.seed(2)

# Number of observations for simulation
n_obs = 200

# Truth data
field_size = (100,100)
target_centre = np.array([62.0,46.0])
target_radius = 15.0

# Observation model
obs_model = sensor_models.logistic_obs
obs_kwargs = {'r':target_radius,'true_pos':0.99,'true_neg':0.99, 'decay_rate':0.5}

# Start state (location and heading rad)
start_pose = np.array([18.0, 23.0, np.pi/2])

# Prior sampler
mcsamples = 5000

# Other constants
obs_symbols = ['r^','go']

# Plot sensor curve
xx = np.linspace(0,3*target_radius,101)
yy = [obs_model(x,True,c=0,**obs_kwargs) for x in xx]
fobs,axobs = plt.subplots()
axobs.plot(xx,yy)
axobs.set_ylim(0,1.0)
axobs.set_ylabel(r'$P(z(r) = T)$'); axobs.set_xlabel('Range, $r$')
fobs.show()

# Setup vehicle belief
glider_motion = yaw_rate_motion(max_yaw=np.pi, speed=4.0, n_yaws=6, n_points=21)
vehicle_belief = belief_state.belief(field_size, obs_model, obs_kwargs,motion_model=glider_motion)

h_fig, h_ax = plt.subplots() #1,2, sharex=True, sharey=True)
h_cpos, = h_ax.plot([],[],'yo',fillstyle='full')
h_pc = h_ax.imshow(np.zeros(field_size), origin='lower',vmin=0,animated=True)
h_tgt, = h_ax.plot(target_centre[0],target_centre[1],'wx',mew=2,ms=10)
h_start, = h_ax.plot(start_pose[0],start_pose[1],'^',color='orange',ms=8,fillstyle='full')
h_obsF, = h_ax.plot([],[],obs_symbols[0])
h_obsT, = h_ax.plot([],[],obs_symbols[1])
h_poss, = h_ax.plot([],[],'k.')
h_path, = h_ax.plot([],[],'w-')
h_ax.set_xlim(-.5, field_size[0]-0.5)
h_ax.set_ylim(-.5, field_size[1]-0.5)
full_path = np.array([start_pose])

def init():
    global full_path
    vehicle_belief.set_current_pose(copy.copy(start_pose))
    vehicle_belief.reset_observations()
    full_path = np.array([start_pose])
    
    xs = vehicle_belief.uniform_prior_sampler(mcsamples)
    vehicle_belief.assign_prior_sample_set(xs)
   
    obs = vehicle_belief.generate_observations([start_pose[0:2]],c=target_centre)
    vehicle_belief.add_observations(obs)
    vehicle_belief.update_pc_map = False
    pc = vehicle_belief.persistent_centre_probability_map()
    h_pc.set_data(pc.transpose())
    h_pc.set_clim([0, pc.max()])
    
    if obs[0][1]:
        h_obsT.set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==True]))
        h_obsF.set_data([],[])
    else:
        h_obsF.set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==False]))
        h_obsT.set_data([],[])
    h_cpos.set_data(start_pose[0],start_pose[1])
    h_path.set_data(full_path[:,0],full_path[:,1])
    return h_pc,h_cpos,h_obsF,h_obsT,h_poss,h_tgt,h_start,h_path

def animate(i):
    global full_path
    opose = vehicle_belief.get_current_pose()
    fposes,Vx,amax = vehicle_belief.select_observation(target_centre)
    full_path = np.append(full_path, glider_motion.get_trajectory(opose,amax),axis=0)
    Vx = Vx-Vx.min()
    h_poss.set_data(fposes[:,0],fposes[:,1]) #,color=plt.cm.jet(Vx/Vx.max()) 
    obs = vehicle_belief.get_observations()
    if obs[-1][1]:
        h_obsT.set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==True]))
    else:
        h_obsF.set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==False]))
    
    #if i % 5 == 0:
    pc = vehicle_belief.persistent_centre_probability_map()
    h_pc.set_data(pc.transpose())
    h_pc.set_clim([0, pc.max()])    
    
    cpos = vehicle_belief.get_current_pose()
    h_cpos.set_data(cpos[0],cpos[1])   
    h_path.set_data(full_path[:,0],full_path[:,1])
    print "i = {0}/{1}".format(i+1,n_obs)
    return h_pc,h_cpos,h_obsF,h_obsT,h_poss,h_tgt,h_start,h_path

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = True, repeat = False)
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
