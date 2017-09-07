import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
import motion_models
import belief_state
import copy
plt.style.use('ggplot')
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
randseed = 1

# Number of observations for simulation
n_obs = 100

# Truth data
field_size = (100,100)
target_centre = np.array([62.0,46.0])
target_radius = 15.0

# KLD tree search depth
kld_depth = 2

# Target range for target finder (50% probability mass in 1% of area near true target)
target_range = np.sqrt((field_size[0]*field_size[1]/100.0)/np.pi)

# Observation model
obs_model = sensor_models.BinaryLogisticObs(r=target_radius, true_pos=0.9, true_neg=0.9, decay_rate=0.35)

# Start state (location and heading rad)
start_pose = np.array([18.0, 23.0, np.pi/2])

# Prior sampler
mcsamples = 3000

# Other constants
obs_symbols = ['r^','go']

# Plot sensor curve
fobs,axobs = plt.subplots()
obs_model.plot(axobs)
fobs.show()

# World model
world = belief_state.World(*field_size, target_location=target_centre)

# Setup vehicles
glider_motion = motion_models.yaw_rate_motion(max_yaw=np.pi/2.0, speed=4.0, n_yaws=5, n_points=21)
vehicle = belief_state.Vehicle(world, glider_motion, obs_model.likelihood, start_state=start_pose)

p_range = belief_state.TargetFinder(target_centre, vehicle.belief, target_range)

h_fig, h_ax = plt.subplots() #1,2, sharex=True, sharey=True)

def init():
    np.random.seed(randseed)
    
    vehicle.reset()
    
    # Generate MC samples
    vehicle.belief.uniform_prior_sampler(mcsamples, set_samples=True)
    
    # Generate observations
    obs = vehicle.generate_observations([vehicle.get_current_pose()])
    vehicle.add_observations(obs)
    vehicle.belief.update_pc_map = False
    
    # Build KLD likelihood tree
    vehicle.build_likelihood_tree(kld_depth)
    
    # Reset target within range calculator
    p_range.reset()
    
    # Generate persistent probability map
    vehicle.belief.persistent_centre_probability_map()
    
    vehicle.setup_plot(h_ax, tree_depth = kld_depth)
    vehicle.update_plot()
    return vehicle.get_artists()

def animate(i):
    vehicle.kld_select_obs(kld_depth)
    vehicle.update_plot()
    print "i = {0}/{1}, p_range={2}".format(i+1,n_obs, p_range.prob_mass_in_range())
    return vehicle.get_artists()

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = True, repeat = False)
#ani.save('../vid/temp.ogv', writer = 'avconv', fps=5, codec='libtheora') #extra_args=['-vcodec', 'libx264'])
h_fig.show()

#h_mov = []
#f2,a2 = plt.subplots()
#vehicle.setup_plot(
#for i in range(10):
#    vehicle.kld_select_obs(kld_depth)
#    h_mov.append(vehicle.update_plot())
#ani2 = animation.ArtistAnimation(h_fig, h_mov)

## SCRAP
    # Plotting
    #vehicle.h_artists['pc'].set_data(pc.transpose())
    #vehicle.h_artists['pc'].set_clim([0, pc.max()])
    # h_poss.set_offsets(start_offsets)
    
    #if obs[0][1]:
    #    vehicle.h_artists['obsT'].set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==True]))
    #    vehicle.h_artists['obsF'].set_data([],[])
    #else:
    #    vehicle.h_artists['obsF'].set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==False]))
    #    vehicle.h_artists['obsT'].set_data([],[])
    #h_cpos.set_data(start_pose[0],start_pose[1])
    #h_path.set_data(full_path[:,0],full_path[:,1])
    
    #return h_pc,h_cpos,h_obsF,h_obsT,h_poss,h_tgt,h_start,h_path    

    #global full_path
    #opose = vehicle_belief.get_current_pose()
    
    #fposes,Vx,amax = vehicle_belief.select_observation(target_centre)
    #fposes,Vx,amax = vehicle_belief.kld_select_obs(kld_depth,target_centre)
    #full_path = np.append(full_path, glider_motion.get_trajectory(opose,amax),axis=0)
    
    #Vx = Vx-Vx.min()
    #h_poss.set_data(fposes[:,0],fposes[:,1]) #,color=plt.cm.jet(Vx/Vx.max()) 
    #h_poss.set_offsets(fposes[:,0:2])
    #h_poss.set_array(Vx)
    
    #obs = vehicle_belief.get_observations()
    #if obs[-1][1]:
    #    h_obsT.set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==True]))
    #else:
    #    h_obsF.set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==False]))
    
    #if i % 5 == 0:
    #pc = vehicle_belief.persistent_centre_probability_map()
    #h_pc.set_data(pc.transpose())
    #h_pc.set_clim([0, pc.max()])    
    #
    #cpos = vehicle_belief.get_current_pose()
    #h_cpos.set_data(cpos[0],cpos[1])   
    #h_path.set_data(full_path[:,0],full_path[:,1])

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
