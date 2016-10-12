import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
import motion_models
import belief_state
import copy
plt.style.use('ggplot')
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
randseed = 1

# Number of observations for simulation
n_obs = 120

# Truth data
field_size = (100,100)
target_centre = np.array([62.0,46.0])
target_radius = 15.0

# KLD tree search depth
kld_depth = 1

# Target range for target finder (50% probability mass in 1% of area near true target)
target_range = np.sqrt((field_size[0]*field_size[1]/100.0)/np.pi)

# Observation model
obs_model = sensor_models.logistic_obs
obs_kwargs = {'r':target_radius,'true_pos':0.9,'true_neg':0.9, 'decay_rate':0.35}

# Start state (location and heading rad)
start_pose = np.array([18.0, 23.0, np.pi/2])

# Prior sampler
mcsamples = 3000

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
glider_motion = motion_models.yaw_rate_motion(max_yaw=np.pi/2.0, speed=4.0, n_yaws=5, n_points=21)
vehicle_belief = belief_state.Belief(field_size, obs_model, obs_kwargs,motion_model=glider_motion)
start_offsets = glider_motion.get_leaf_points(start_pose,depth=kld_depth)[:,0:2]

h_fig, h_ax = plt.subplots() #1,2, sharex=True, sharey=True)
h_cpos, = h_ax.plot([],[],'yo',fillstyle='full')
h_pc = h_ax.imshow(np.zeros(field_size), origin='lower',vmin=0,animated=True)
h_tgt, = h_ax.plot(target_centre[0],target_centre[1],'wx',mew=2,ms=10)
h_start, = h_ax.plot(start_pose[0],start_pose[1],'^',color='orange',ms=8,fillstyle='full')
h_obsF, = h_ax.plot([],[],obs_symbols[0])
h_obsT, = h_ax.plot([],[],obs_symbols[1])
#h_poss, = h_ax.plot([],[],'k.')
h_poss = h_ax.scatter(start_offsets[:,0],start_offsets[:,1],20)
h_path, = h_ax.plot([],[],'w-')
h_ax.set_xlim(-.5, field_size[0]-0.5)
h_ax.set_ylim(-.5, field_size[1]-0.5)
full_path = np.array([start_pose])

p_range = belief_state.TargetFinder(target_centre, vehicle_belief, target_range)

def init():
    global full_path
    np.random.seed(randseed)
    
    vehicle_belief.set_current_pose(copy.copy(start_pose))
    vehicle_belief.reset_observations()
    full_path = np.array([start_pose])
    
    # Generate MC samples
    vehicle_belief.uniform_prior_sampler(mcsamples, set_samples=True)
   
    obs = vehicle_belief.generate_observations([start_pose[0:2]],c=target_centre)
    vehicle_belief.add_observations(obs)
    vehicle_belief.update_pc_map = False
    vehicle_belief.build_likelihood_tree(kld_depth)
    p_range.reset()
    pc = vehicle_belief.persistent_centre_probability_map()
    h_pc.set_data(pc.transpose())
    h_pc.set_clim([0, pc.max()])
    h_poss.set_offsets(start_offsets)
    
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
    
    #fposes,Vx,amax = vehicle_belief.select_observation(target_centre)
    fposes,Vx,amax = vehicle_belief.kld_select_obs(kld_depth,target_centre)
    full_path = np.append(full_path, glider_motion.get_trajectory(opose,amax),axis=0)
    Vx = Vx-Vx.min()
    #h_poss.set_data(fposes[:,0],fposes[:,1]) #,color=plt.cm.jet(Vx/Vx.max()) 
    h_poss.set_offsets(fposes[:,0:2])
    h_poss.set_array(Vx)
    
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
    print "i = {0}/{1}, p_range={2}".format(i+1,n_obs, p_range.prob_mass_in_range())
    return h_pc,h_cpos,h_obsF,h_obsT,h_poss,h_tgt,h_start,h_path

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = True, repeat = False)
#ani.save('../vid/single_tracking3.mp4', writer = 'avconv', fps=5, bitrate=15000, codec='libx264') #extra_args=['-vcodec', 'libx264'])
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
