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
randseed = 2

n_obs = 120         # Number of observations for simulation
n_vehicles = 3      # Number of vehicles

# Truth data
field_size = (100,100)
target_centre = np.array([25.0,75.0])
target_radius = 15.0

# Observation model
obs_model = sensor_models.logistic_obs
obs_kwargs = {'r':target_radius,'true_pos':0.9,'true_neg':0.9, 'decay_rate':0.35}
#obs_model = sensor_models.step_obs
#obs_kwargs = {'r':target_radius,'true_pos':0.9,'true_neg':0.9}

# Start state (location and heading rad)
start_rand = np.random.rand(n_vehicles,3)
start_pose = np.array([[x*field_size[0],y*field_size[1],(z-0.5)*2*np.pi] for x,y,z in start_rand])
#start_pose = np.array([[18.0, 23.0, np.pi/2],[75.0, 75.0, -np.pi/2]])

# Prior sampler
mcsamples = 2000

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

# Setup vehicle belief
glider_motion = yaw_rate_motion(max_yaw=np.pi, speed=4.0, n_yaws=6, n_points=21)
vehicle_beliefs = [belief_state.belief_unshared(field_size, obs_model, obs_kwargs,motion_model=glider_motion) for i in range(n_vehicles)]

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
full_path=[[] for i in range(n_vehicles)]
h_art = [[] for i in range(n_vehicles)]
for i in range(n_vehicles):
    h_art[i].append(h_ax[i].imshow(np.zeros(field_size), origin='lower',vmin=0,animated=True)) #0 Image
    h_art[i].extend(h_ax[i].plot(target_centre[0],target_centre[1],'wx',mew=2,ms=10)) #1 Target location
    h_art[i].extend(h_ax[i].plot(start_pose[i][0],start_pose[i][1],'yo',mew=0.5)) #2 Start location
    h_art[i].extend(h_ax[i].plot([],[],'^',color='darksalmon',mec='w',mew=0,ms=5)) #3 Shared False
    h_art[i].extend(h_ax[i].plot([],[],'o',color='darkseagreen',mec='w',mew=0,ms=5)) #4 Shared True
    h_art[i].extend(h_ax[i].plot([],[],'w-',lw=2)) #5 Trajectory
    h_art[i].extend(h_ax[i].plot([],[],'o',color='gold',ms=8,mew=0)) #6 Current position
    h_art[i].extend(h_ax[i].plot([],[],obs_symbols[0],mew=0.5,mec='w',ms=6.5)) #7 Local False
    h_art[i].extend(h_ax[i].plot([],[],obs_symbols[1],mew=0.5,mec='w',ms=6.5)) #8 Local True
    h_art[i].extend(h_ax[i].plot([],[],'k.')) #9 Possible moves
    h_ax[i].set_xlim(-.5, field_size[0]-0.5)
    h_ax[i].set_ylim(-.5, field_size[1]-0.5)    
    full_path[i].append(start_pose[i])


def init():
    global full_path,curr_dkl,last_share
    curr_dkl = 0.1*max_dkl
    last_share = 0
    np.random.seed(randseed)
    full_path=[[] for i in range(n_vehicles)]
    for i,vehicle_belief in enumerate(vehicle_beliefs):
        vehicle_belief.set_current_pose(copy.copy(start_pose[i]))
        vehicle_belief.reset_observations()
        
        xs = vehicle_belief.uniform_prior_sampler(mcsamples)
        vehicle_belief.assign_prior_sample_set(xs)
   
        obs = vehicle_belief.generate_observations([start_pose[i,0:2]],c=target_centre)
        vehicle_belief.add_observations(obs)
        vehicle_belief.update_pc_map = False
        pc = vehicle_belief.persistent_centre_probability_map()
        h_art[i][0].set_data(pc.transpose())
        h_art[i][0].set_clim([0, pc.max()])
    
        if obs[0][1]:
            h_art[i][8].set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==True]))
            h_art[i][7].set_data([],[])
        else:
            h_art[i][7].set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs if zz==False]))
            h_art[i][8].set_data([],[])
        h_art[i][6].set_data(start_pose[i][0],start_pose[i][1])
        h_art[i][3].set_data([],[])
        h_art[i][4].set_data([],[])
    return [item for sublist in h_art for item in sublist]

vehicle_set = set(range(n_vehicles))
def animate(i):
    global full_path, curr_dkl, last_share, current_wait
    if current_wait == share_wait:
        Fobs = [xx for xx,zz in vehicle_beliefs[0].observations if zz==False]
        Tobs = [xx for xx,zz in vehicle_beliefs[0].observations if zz==True]
        for hh in h_art:
            hh[3].set_data(*zip(*Fobs))
            hh[4].set_data(*zip(*Tobs))
        current_wait += 1
    elif current_wait < share_wait:
        current_wait += 1
        return [item for sublist in h_art for item in sublist]
    
    dd = [dkl_map(vb) for vb in vehicle_beliefs]
    vm = np.argmax(dd)
    nobs = len(vehicle_beliefs[0].observations)
        
    if dd[vm] > curr_dkl:
        # We share :)
        for ii in range(n_vehicles):
            other_vehicles = vehicle_set - {ii}
            for jj in other_vehicles:
                vehicle_beliefs[ii].add_observations(vehicle_beliefs[jj].observations[last_share:nobs],vehicle_beliefs[jj].unshared_pIgc,vehicle_beliefs[jj].unshared_pIgc_map)
        for vb in vehicle_beliefs:
            vb.reset_unshared()
        last_share = len(vehicle_beliefs[0].observations)
        print "Shared at {0}".format(last_share)
        curr_dkl = dd[vm]+delta_dkl*max_dkl
        current_wait = 0
        return [item for sublist in h_art for item in sublist]
    
    else:
        for hh,vehicle_belief,fp in zip(h_art,vehicle_beliefs,full_path):
            opose = vehicle_belief.get_current_pose()
            fposes,Vx,amax = vehicle_belief.select_observation(target_centre)
            fp.extend(glider_motion.get_trajectory(opose,amax))
            Vx = Vx-Vx.min()
            hh[9].set_data(fposes[:,0],fposes[:,1]) 
            obs = vehicle_belief.get_observations()
            if obs[-1][1]:
                hh[8].set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs[last_share:] if zz==True]))
            else:
                hh[7].set_data(*zip(*[[xx[0],xx[1]] for xx,zz in obs[last_share:] if zz==False]))
        
            cpos = vehicle_belief.get_current_pose()
            hh[6].set_data(cpos[0],cpos[1])
            pp = np.array(fp)
            hh[5].set_data(pp[:,0],pp[:,1])
    
    pcmax = 0.0
    for hh,vb in zip(h_art,vehicle_beliefs):
        pc = vb.persistent_centre_probability_map()
        intpc = pc.sum()
        hh[0].set_data(pc.transpose()/intpc)
        pcmax = max(pcmax,pc.max()/intpc)
    for hh in h_art:
        hh[0].set_clim([0, pcmax])    
    
    print "i = {0}/{1}".format(i+1,n_obs)
    return [item for sublist in h_art for item in sublist]

ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = True, repeat = False)
#ani.save('vid/multiple_tracking.mp4', writer = 'avconv', fps=3, bitrate=5000, codec='libx264')
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
