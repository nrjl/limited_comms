import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
from motion_models import yaw_rate_motion
import belief_state
import copy
import pickle

PLOT_ON = False

class MultirobotSearch(object):
    
    def __init__(self, field_size, n_robots, obs_model, obs_kwargs, motion_model, mcsamples, kld_depth=1, delta_dkl = 0.2, randseed=0):
        # Setup vehicle belief
        self.world = belief_state.World(*field_size)
        self.n_vehicles = n_robots
        self.vehicle_set = set(range(self.n_vehicles))
        self.mcsamples = mcsamples
        self.kld_depth = kld_depth
        self.randseed = randseed
        
        self.vehicles = [belief_state.Vehicle(self.world, motion_model, 
            obs_model, obs_kwargs, unshared=True) for i in self.vehicle_set]
        #self.vehicle_beliefs = [belief_state.BeliefUnshared(
        #    field_size, obs_model, obs_kwargs,
        #    motion_model=motion_model) for i in self.vehicle_set]
        
        # D_KL bounds
        self.max_dkl = np.log(np.prod(self.world.get_size()))
        self.delta_dkl = delta_dkl
        
        # Figure handles
        #self.h_fig, self.h_ax = plt.subplots(1,self.n_vehicles) #, sharex=True, sharey=True)
        #self.h_fig.set_size_inches(5*self.n_vehicles,5,forward=True)
            
    def dkl_mc(self, vb):
        pcgI = vb.pIgc/vb.mc_p_I()*vb.p_c(0)
        dkl = pcgI*np.log(pcgI/vb.p_c(0))
        return dkl.sum()/pcgI.sum()
        
    def sync_beliefs(self):
        cobs = self.vehicles[-1].get_observations()
        cpIgc = self.vehicles[-1].belief.pIgc
        cpIgc_map = self.vehicles[-1].belief.pIgc_map
        for v in self.vehicles[:-1]:
            cobs.extend(v.get_observations()[self.last_share:])
            cpIgc *= v.belief.unshared_pIgc
            cpIgc_map *= v.belief.unshared_pIgc_map
        for v in self.vehicles:
            v.belief.observations = cobs
            v.belief.pIgc = cpIgc
            v.belief.pIgc_map = cpIgc_map
            v.belief.reset_unshared()

    def reset(self):
        self.curr_dkl = self.delta_dkl*self.max_dkl
        self.last_share = 0
        self.number_of_shares = 0
        self.randseed += 1
        np.random.seed(self.randseed)
        
        # Target location
        self.world.set_target_location(np.random.rand(2)*np.array(self.world.get_size()))
        
        # Start state (location and heading rad)
        start_rand = np.random.rand(self.n_vehicles,3)
        start_poses = np.array([[x*self.world.get_size()[0],y*self.world.get_size()[1],(z-0.5)*2*np.pi] for x,y,z in start_rand])
        
        xs = self.vehicles[0].belief.uniform_prior_sampler(mcsamples)
        for v,sp in zip(self.vehicles, start_poses):
            v.reset(copy.copy(sp))
            v.belief.assign_prior_sample_set(xs)
            
            obs = v.generate_observations([v.get_current_pose()[0:2]])
            v.add_observations(obs)
            v.build_likelihood_tree(self.kld_depth)           
    
    def step(self):
        dd = [self.dkl_mc(v.belief) for v in self.vehicles]
        vm = np.argmax(dd)
        nobs = len(self.vehicles[0].get_observations())
            
        if dd[vm] > self.curr_dkl:
            # We share :)
            for ii,v in enumerate(self.vehicles):
                other_vehicles = [self.vehicles[ovb] for ovb in (self.vehicle_set-{ii})]
                for ov in other_vehicles:
                    v.add_observations(ov.get_observations()[self.last_share:nobs],ov.belief.unshared_pIgc)
            for v in self.vehicles:
                v.belief.reset_unshared()
            self.last_share = len(self.vehicles[0].get_observations())
            print "Total shared observations: {0}".format(self.last_share)
            self.curr_dkl = dd[vm]+self.delta_dkl*self.max_dkl
            self.number_of_shares += 1
            return True
        
        else:
            for v in self.vehicles:
                v.kld_select_obs(self.kld_depth)
        return False

n_trials = 1      # Number of trials
max_obs = 200       # Max. observations for simulation
n_robots = 3        # Number of vehicles
share_cost = 0      # Cost of sharing (in # observations)
randseed = 80

# Truth data
field_size = (100,100)
target_radius = 15.0
kld_tree_depth = 2
delta_dkl = 0.1

# Observation model
sensor_fun = sensor_models.logistic_obs
sensor_kwargs = {'r':target_radius,'true_pos':0.9,'true_neg':0.9, 'decay_rate':0.35}

# Motion model
glider_motion = yaw_rate_motion(max_yaw=np.pi, speed=4.0, n_yaws=6, n_points=21)

# Termination (90% probability mass in 1% of area near true target)
pmass_range = np.sqrt((field_size[0]*field_size[1]/100.0)/np.pi)
pmass_prob = 0.9

# Prior sampler
mcsamples = 2000

# Other constants
obs_symbols = ['r^','go']

# Plot sensor curve
if PLOT_ON:
    plt.ion()
    
    xx = np.linspace(0,3*target_radius,101)
    yy = [sensor_fun(x,True,c=0,**sensor_kwargs) for x in xx]
    fobs,axobs = plt.subplots()
    axobs.plot(xx,yy)
    axobs.set_ylim(0,1.0)
    axobs.set_ylabel(r'$P(z(r) = T)$'); axobs.set_xlabel('Range, $r$')

    fpmass,axpmass = plt.subplots()
    hpmass = []
    for ii in range(n_robots):
        hpmass.extend(axpmass.plot(0,0))
    axpmass.set_xlim([0,max_obs])
    axpmass.set_ylim([0,1.0])

# Create multirobot searcher instance
multirobot_runner = MultirobotSearch(field_size, n_robots, sensor_fun, sensor_kwargs, 
    glider_motion, mcsamples, kld_tree_depth, delta_dkl, randseed=randseed)
trial_steps = []
n_shares = []

for trial_num in range(n_trials):
    multirobot_runner.reset()
    target_pmass = [belief_state.TargetFinder(multirobot_runner.world.get_target_location(), v.belief, pmass_range) 
        for v in multirobot_runner.vehicles]
    
    ii = 0
    pmass = [[] for jj in range(n_robots)]
    target_found = False
    while (ii < max_obs) and not target_found:
        if multirobot_runner.step() == True:
            ii += share_cost
        else:
            ii += 1
        #print "i = {0}".format(ii)

        for jj,pm in enumerate(pmass):
            current_pmass = target_pmass[jj].prob_mass_in_range()
            pm.append(current_pmass)
            if PLOT_ON: hpmass[jj].set_data(range(len(pm)),pm)
            target_found = target_found or (current_pmass>pmass_prob)
        if PLOT_ON:
            fpmass.canvas.draw()
            fpmass.canvas.flush_events()

    trial_steps.append(ii)
    n_shares.append(multirobot_runner.number_of_shares)
    print "Trial {0} complete. Target found in {1} steps.".format(trial_num, ii)
    
with open('../data/temp.pkl', 'wb') as fh:
    pickle.dump(trial_steps, fh)
    pickle.dump(n_shares, fh)

    
    #ani = animation.FuncAnimation(h_fig, animate, init_func = init, frames = n_obs, interval = 100, blit = True, repeat = False)
