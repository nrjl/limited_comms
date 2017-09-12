import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
import belief_state
import prm
import random

plt.rc('text.latex', preamble='\usepackage{amsmath},\usepackage{amssymb}')
randseed = 0

n_obs = 120     # Number of observations for simulation
n_vehicles = 3  # Number of vehicles

# Truth data
field_size = (100, 100)
target_centre = np.array([62.0, 46.0])
target_radius = 15.0

kld_depth = 2

# Observation model
sensor = sensor_models.BinaryLogisticObs(r=target_radius, true_pos=0.9, true_neg=0.9, decay_rate=0.35)

# PRM graph
prm_nodes = 600
roadmap = prm.PRM([[0.0, field_size[0]], [0, field_size[1]]], prm_nodes, type='kPRM*')
print roadmap
print roadmap.G

# Prior sampler
mcsamples = 1000

# Other constants
obs_symbols = ['r^', 'go']

# Share parameters
max_dkl = np.log(field_size[0]*field_size[1])
curr_dkl = 0.05*max_dkl
delta_dkl = 0.1
last_share = 0

# How long to wait when there's a share frame
share_wait = 0
current_wait = share_wait + 1

# Plot sensor curve
fobs, axobs = plt.subplots()
sensor.plot(axobs)
fobs.show()

# World model
world = belief_state.World(*field_size, target_location=target_centre)

# Motion model
vehicle_motion = prm.GraphMotion(roadmap.G)

# Start state
start_state = random.choice(roadmap.G.V.keys())

# Setup vehicles
vehicles = [prm.GraphVehicle(world, vehicle_motion, sensor, start_state, unshared=True) for i in range(n_vehicles)]

h_fig, h_ax = plt.subplots(1, n_vehicles)  # , sharex=True, sharey=True)
h_fig.set_size_inches(5 * n_vehicles, 5, forward=True)


def get_all_artists(vv):
    all_artists = []
    for v in vv:
        all_artists.extend(v.get_artists())
    return all_artists

def arrange_meeting(prm, start_id, max_cost, n_robots):
    path_groups = prm.pull_path_groups(start_id, max_cost, n_robots)

def init():
    global curr_dkl, last_share
    curr_dkl = 0.1 * max_dkl
    last_share = 0

    # np.random.seed(randseed)

    # Generate MC samples
    xs = vehicles[0].belief.uniform_prior_sampler(mcsamples)

    # Generate start observation
    obs = vehicles[0].generate_observations([vehicles[0].get_current_pose()], set_obs=False)

    for vehicle, hv in zip(vehicles, h_ax):
        vehicle.reset(start_state, xs=xs)

        # Set observations
        vehicle.add_observations(obs)
        vehicle.belief.update_pc_map = False

        # Build KLD likelihood tree
        vehicle.build_likelihood_tree(kld_depth)

        # Generate persistent probability map (for plotting)
        vehicle.belief.persistent_centre_probability_map()

        vehicle.setup_plot(hv, tree_depth=kld_depth)
        vehicle.update_plot()

    arrange_meeting(roadmap, start_state, max_cost=20.0, n_robots=n_vehicles)

    return get_all_artists(vehicles)


def animate(i):
    global curr_dkl, last_share, current_wait
    if current_wait == share_wait:
        Fobs = [xx for xx, zz in vehicles[0].belief.get_observations() if zz == False]
        Tobs = [xx for xx, zz in vehicles[0].belief.get_observations() if zz == True]
        for vehicle in vehicles:
            vehicle.h_artists['shared_obsF'].set_data([o[0] for o in Fobs], [o[1] for o in Fobs])
            vehicle.h_artists['shared_obsT'].set_data([o[0] for o in Tobs], [o[1] for o in Tobs])
        current_wait += 1
    elif current_wait < share_wait:
        current_wait += 1
        return get_all_artists(vehicles)

    dd = [vehicle.belief.dkl_map() for vehicle in vehicles]
    vm = np.argmax(dd)

    if dd[vm] > curr_dkl:
        # We share :)
        last_share = belief_state.share_beliefs(vehicles, last_share)
        print "Shared at {0}".format(last_share)
        curr_dkl = dd[vm] + delta_dkl * max_dkl
        current_wait = 0
        return get_all_artists(vehicles)

    else:
        for vehicle in vehicles:
            vehicle.kld_select_obs(kld_depth)
            vehicle.update_plot()

    for vehicle in vehicles:
        vehicle.belief.persistent_centre_probability_map()

    pcmax = max([v.h_artists['pc'].get_clim()[1] for v in vehicles])
    for v in vehicles:
        v.h_artists['pc'].set_clim([0, pcmax])

    print "i = {0}/{1}".format(i + 1, n_obs)
    return get_all_artists(vehicles)


ani = animation.FuncAnimation(h_fig, animate, init_func=init, frames=n_obs, interval=100, blit=True, repeat=False)
# ani.save('../vid/temp.ogv', writer = 'avconv', fps=3, bitrate=5000, codec='libtheora')
h_fig.show()