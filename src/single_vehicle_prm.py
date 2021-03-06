# Similar to single vehicle search but use PRM instead of motion model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
import belief_state
import random
import prm

plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)
randseed = 100
np.random.seed(randseed)

# Number of observations for simulation
n_obs = 10

# Truth data
field_size = (100, 80)
target_centre = np.array([62.0, 46.0])
target_radius = 15.0

# KLD tree search depth
kld_depth = 2

# PRM graph
prm_nodes = 600
roadmap = prm.PRM([[0.0, field_size[0]], [0, field_size[1]]], prm_nodes, type='kPRM*')
print roadmap
print roadmap.G
if roadmap.G.get_number_edges() < 5000:
    fh, ah = plt.subplots()
    label_nodes = (prm_nodes <= 100)
    roadmap.plot_PRM(ah, label_nodes=label_nodes)
    fh.show()
all_locations = roadmap.G.get_all_locations()

# Target range for target finder (50% probability mass in 1% of area near true target)
target_range = np.sqrt((field_size[0] * field_size[1] / 100.0) / np.pi)

# Observation model
sensor = sensor_models.BinaryLogisticObs(r=target_radius, true_pos=0.9, true_neg=0.9, decay_rate=0.35)

# Start state
start_state = np.random.choice(roadmap.G.V.keys())

# Prior sampler
mcsamples = 3000

# Other constants
obs_symbols = ['r^', 'go']

# Plot sensor curve
fobs,axobs = plt.subplots()
sensor.plot(axobs)
fobs.show()

# World model
world = belief_state.World(*field_size, target_location=target_centre)

# Setup vehicles
vehicle_motion = prm.GraphMotion(roadmap.G)
vehicle = prm.GraphVehicle(world, vehicle_motion, sensor, start_state)

p_range = belief_state.TargetFinder(target_centre, vehicle.belief, target_range)

h_fig, h_ax = plt.subplots()  # 1,2, sharex=True, sharey=True)


def init():
    np.random.seed(randseed)

    # Generate MC samples
    xs = vehicle.belief.uniform_prior_sampler(mcsamples)

    vehicle.reset(xs = xs)

    # Generate observations
    vehicle.generate_observations([vehicle.get_current_pose()], set_obs=True)
    vehicle.belief.update_pc_map = False

    # Build KLD likelihood tree
    # vehicle.build_likelihood_tree(kld_depth)

    # Reset target within range calculator
    p_range.reset()

    # Generate persistent probability map
    vehicle.belief.persistent_centre_probability_map()

    vehicle.setup_plot(h_ax, tree_depth=kld_depth)
    vehicle.h_G, = h_ax.plot(all_locations[:,0], all_locations[:,1], 'k.', ms=3.0)
    vehicle.update_plot()
    return vehicle.get_artists() + (vehicle.h_G,)


def animate(i):
    vehicle.kld_select_obs(kld_depth)
    vehicle.update_plot()
    print "i = {0}/{1}, p_range={2}".format(i + 1, n_obs, p_range.prob_mass_in_range())
    return vehicle.get_artists() + (vehicle.h_G,)


ani = animation.FuncAnimation(h_fig, animate, init_func=init, frames=n_obs, interval=100, blit=True, repeat=False)
# ani.save('../vid/temp.ogv', writer = 'avconv', fps=5, codec='libtheora') #extra_args=['-vcodec', 'libx264'])
h_fig.show()


