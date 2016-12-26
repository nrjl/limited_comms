import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
from motion_models import yaw_rate_motion
import belief_state
import copy
#plt.style.use('ggplot')
#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
randseed = 0

n_obs = 120         # Number of observations for simulation
n_vehicles = 3      # Number of vehicles

# Truth data
field_size = (100,100)
target_centre = np.array([25.0,75.0])
target_radius = 15.0

kld_depth = 2

# Observation model
obs_model = sensor_models.logistic_obs
obs_kwargs = {'r':target_radius,'true_pos':0.8,'true_neg':0.8, 'decay_rate':0.35}
#obs_model = sensor_models.step_obs
#obs_kwargs = {'r':target_radius,'true_pos':0.9,'true_neg':0.9}

# Start state (location and heading rad)
start_rand = np.random.rand(n_vehicles,3)
start_poses = np.array([[x*field_size[0],y*field_size[1],(z-0.5)*2*np.pi] for x,y,z in start_rand])
#start_poses = np.array([[18.0, 23.0, np.pi/2],[75.0, 75.0, -np.pi/2]])

# Prior sampler
mcsamples = 1000

# Other constants
obs_symbols = ['r^','go']

# D_KL bounds
max_dkl = np.log(field_size[0]*field_size[1])
delta_dkl = 0.2
curr_dkl = 0.1*max_dkl
last_share = 0

# How long to wait when there's a share frame
share_wait = 5
current_wait = share_wait+1

# Plot sensor curve
xx = np.linspace(0,3*target_radius,101)
yy = [obs_model(x,True,c=0,**obs_kwargs) for x in xx]
fobs,axobs = plt.subplots()
axobs.plot(xx,yy)
axobs.set_ylim(0,1.0)
axobs.set_ylabel(r'$P(z(r) = T)$'); axobs.set_xlabel('Range, $r$')
fobs.show()

# World model
world = belief_state.World(*field_size, target_location=target_centre)

# Setup vehicle belief
glider_motion = yaw_rate_motion(max_yaw=np.pi, speed=4.0, n_yaws=6, n_points=21)
vehicles = [belief_state.Vehicle(world, glider_motion, obs_model, obs_kwargs, start_pose, unshared=True) for start_pose in start_poses]
#vehicle_beliefs = [belief_state.BeliefUnshared(field_size, obs_model, obs_kwargs,motion_model=glider_motion) for i in range(n_vehicles)]
#start_offsets = [glider_motion.get_leaf_points(sp,depth=kld_depth)[:,0:2] for sp in start_poses]

def dkl_map(vb):
    pcgI = vb.persistent_centre_probability_map()
    dkl = pcgI*np.log(pcgI/vb.p_c(0))
    return dkl.sum()/pcgI.sum()
    
def sync_beliefs(vbs,last_share):
    cobs = vbs[-1].observations
    cpIgc = vbs[-1].pIgc
    cpIgc_map = vbs[-1].pIgc_map
    for ii in range(n_vehicles-1):
        cobs.extend(vbs[ii].observations[last_share:])
        cpIgc *= vbs[ii].unshared_pIgc
        cpIgc_map *= vbs[ii].unshared_pIgc_map
    for vb in vbs:
        vb.observations = cobs
        vb.pIgc = cpIgc
        vb.pIgc_map = cpIgc_map
        vb.reset_unshared()

h_fig, h_ax = plt.subplots(1,n_vehicles) #, sharex=True, sharey=True)
h_fig.set_size_inches(5*n_vehicles,5,forward=True)

def get_all_artists(vv):
    all_artists = []
    for v in vv:
        all_artists.extend(v.get_artists())
    return all_artists

def init():
    global curr_dkl,last_share
    curr_dkl = 0.1*max_dkl
    last_share = 0
    np.random.seed(randseed)
    
    # Generate MC samples
    xs = vehicles[0].belief.uniform_prior_sampler(mcsamples)
    
    for vehicle, hv in zip(vehicles, h_ax):
        vehicle.reset()
        vehicle.belief.assign_prior_sample_set(xs)
        
        # Generate observations
        obs = vehicle.generate_observations([vehicle.get_current_pose()[0:2]],c=target_centre)
        vehicle.add_observations(obs)
        vehicle.belief.update_pc_map = False
        
        # Build KLD likelihood tree
        vehicle.build_likelihood_tree(kld_depth)
        
        # Generate persistent probability map (for plotting)
        vehicle.belief.persistent_centre_probability_map()
        
        vehicle.setup_plot(hv, tree_depth = kld_depth)
        vehicle.update_plot()
    
    return get_all_artists(vehicles)
    
vehicle_set = set(range(n_vehicles))
def animate(i):
    global curr_dkl, last_share, current_wait
    if current_wait == share_wait:
        Fobs = [xx for xx,zz in vehicles[0].belief.get_observations() if zz==False]
        Tobs = [xx for xx,zz in vehicles[0].belief.get_observations() if zz==True]
        for vehicle in vehicles:
            vehicle.h_artists['shared_obsF'].set_data(*zip(*Fobs))
            vehicle.h_artists['shared_obsT'].set_data(*zip(*Tobs))
        current_wait += 1
    elif current_wait < share_wait:
        current_wait += 1
        return get_all_artists(vehicles)
    
    dd = [dkl_map(vehicle.belief) for vehicle in vehicles]
    vm = np.argmax(dd)
    nobs = len(vehicles[0].get_observations())
        
    if dd[vm] > curr_dkl:
        # We share :)
        for ii in range(n_vehicles):
            other_vehicles = vehicle_set - {ii}
            for jj in other_vehicles:
                vehicles[ii].add_observations(vehicles[jj].get_observations()[last_share:nobs],vehicles[jj].belief.unshared_pIgc,vehicles[jj].belief.unshared_pIgc_map)
        for vehicle in vehicles:
            vehicle.belief.reset_unshared()
        last_share = len(vehicles[0].get_observations())
        print "Shared at {0}".format(last_share)
        curr_dkl = dd[vm]+delta_dkl*max_dkl
        current_wait = 0
        return get_all_artists(vehicles)
    
    else:
        for vehicle in vehicles:
            vehicle.kld_select_obs(kld_depth)
            vehicle.update_plot()
    
    pcmax = 0.0
    for vehicle in vehicles:
        pc = vehicle.belief.persistent_centre_probability_map()
        
    
    pcmax = max([v.h_artists['pc'].get_clim()[1] for v in vehicles])
    for v in vehicles:
        v.h_artists['pc'].set_clim([0, pcmax])
    
    #pcmax = 0.0
    #for hh,vb in zip(h_art,vehicle_beliefs):
    #    pc = vb.persistent_centre_probability_map()
    #    intpc = pc.sum()
    #    hh[0].set_data(pc.transpose()/intpc)
    #    pcmax = max(pcmax,pc.max()/intpc)
    #for hh in h_art:
    #    hh[0].set_clim([0, pcmax])    
    
    print "i = {0}/{1}".format(i+1,n_obs)
    return get_all_artists(vehicles)

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = True, repeat = False)
#ani.save('../vid/temp.ogv', writer = 'avconv', fps=3, bitrate=5000, codec='libtheora')
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
