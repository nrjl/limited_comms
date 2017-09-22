import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sensor_models
import belief_state
import prm
import time
import matplotlib.collections as mc
import nice_plot_colors

plt.rc('font',**{'family':'serif','sans-serif':['Computer Modern Roman']})
plt.rc('text', usetex=True)
randseed = 0

np.random.seed(randseed)

n_obs = 100     # Number of observations for simulation
n_vehicles = 3  # Number of vehicles

# Truth data
field_size = (100, 100)
target_centre = np.array([20.0, 20.0])
target_radius = 15.0

kld_depth = 10
max_path_cost = 25.0

# Observation model
sensor = sensor_models.BinaryLogisticObs(r=target_radius, true_pos=0.9, true_neg=0.9, decay_rate=0.35)

# PRM graph
prm_nodes = 400
ts = time.time()
roadmap = prm.PRM([[0.0, field_size[0]], [0, field_size[1]]], prm_nodes, type='kPRM', k=4)
print roadmap
print roadmap.G
print "Roadmap construction took {0}s".format(time.time()-ts)
if roadmap.G.get_number_edges() < 5000:
    fh, ah = plt.subplots()
    label_nodes = (prm_nodes <= 500)
    roadmap.plot_PRM(ah, label_nodes=label_nodes)
    fh.show()

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
start_state = np.random.choice(roadmap.G.V.keys())

v_colors = [nice_plot_colors.darken(nice_plot_colors.lines[i], 2) for i in range(n_vehicles)]

# Setup vehicles
vehicles = [prm.GraphVehicle(world, vehicle_motion, sensor, start_state, vehicle_color=v_colors[i], unshared=True) for i in range(n_vehicles)]

h_fig, h_ax = plt.subplots(1, n_vehicles)  # , sharex=True, sharey=True)
h_fig.set_size_inches(5 * n_vehicles, 5, forward=True)

def line_segments_from_paths(graph, paths):
    lines = []
    colours = []
    for cc, path in enumerate(paths):
        p1 = graph.get_vertex(path[0]).get_location()
        for i in range(1, len(path)):
            p2 = graph.get_vertex(path[i]).get_location()
            lines.append([p1, p2])
            colours.append(nice_plot_colors.lines[cc])
            p1 = p2
    return lines, colours
    # c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])

def get_all_artists(vv):
    all_artists = []
    for v in vv:
        all_artists.extend(v.get_artists())
    return all_artists

class PathsAndValue(object):
    def __init__(self, paths, value):
        self.paths = paths
        self.value = value

def filter_paths(paths, prefix):
    # This is designed to get rid of redundant paths (that visit the same new nodes or a subset of another path
    new_nodes = np.array([set([newnode for newnode in path[1:-1] if newnode not in prefix]) for path in paths])
    keep_path = np.ones(len(new_nodes), dtype='Bool')
    for i, path_i in enumerate(new_nodes):
        for j, path_j in enumerate(new_nodes):
            if i != j and path_i.issuperset(path_j) and keep_path[i]:
                keep_path[j] = False

    return [paths[k] for k in range(len(paths)) if keep_path[k]]

def arrange_meeting(prm, vehicle, start_id, max_cost, n_robots, min_nodes=3):
    t_start = time.time()
    path_groups = prm.pull_path_groups(start_id, max_cost, n_robots=n_robots, min_nodes=min_nodes)
    t_meet = time.time()-t_start
    total_paths = 0
    for group in path_groups:
        total_paths += len(path_groups[group])
    print "{0} path groups found, {1} total paths in {2}s".format(len(path_groups), total_paths, t_meet)

    t_arrange = time.time()
    i_groups = 0

    group_val = {}
    for end_node, path_group in path_groups.iteritems():
        # For each path group (paths sharing common end) find the best set of n_robots paths and associated E(d_{kl})
        t_group = time.time()
        good_paths = []
        best_E_kld = None  # np.zeros(len(paths))
        best_path = tuple()
        paths = path_group.keys()
        path_prefix = (paths[0][-1],)

        while len(good_paths) < n_robots:

            # if len(paths) == n_robots - len(good_paths):
            #     new_nodes = tuple([newnode for newnode in paths[0][1:-1] if newnode not in path_prefix])
            if len(paths) == n_robots - len(good_paths):
                good_paths.extend(paths)
                for path in good_paths:
                    path_prefix = path_prefix + tuple([newnode for newnode in path[1:-1] if newnode not in path_prefix])
                best_E_kld = vehicle.expected_kld(path_prefix, 0, vehicle.belief.kld_likelihood, vehicle.get_mc_likelihood)
                break

            for i, path in enumerate(paths):
                t_paths = time.time()
                new_nodes = tuple([newnode for newnode in path[1:-1] if newnode not in path_prefix])
                if len(new_nodes) == 0 or set(new_nodes).issubset(best_path):
                    E_kld = best_E_kld
                else:
                    shared_path = path_prefix + new_nodes
                    E_kld = vehicle.expected_kld(shared_path, 0, vehicle.belief.kld_likelihood, vehicle.get_mc_likelihood)
                if E_kld > best_E_kld:
                    best_E_kld = E_kld
                    best_path = path
                # print "New nodes {N}, t={t:0.3f}s, E_kld={E:0.4f}".format(N=new_nodes, t=time.time()-t_paths, E=E_kld)

            good_paths.append(best_path)
            path_prefix = path_prefix + tuple([newnode for newnode in best_path[1:-1] if newnode not in path_prefix])
            # print "Path chosen! {0}".format(best_path)
            paths = filter_paths(paths, path_prefix)
        i_groups += 1

        group_val[end_node] = PathsAndValue(good_paths, best_E_kld)
        print "End node {n} ({i}/{j}), E_kld={E}, solved in {t:0.2f}s".format(n=end_node, E = best_E_kld,
                                                                              t=time.time() - t_group,
                                                                              i=i_groups, j=len(path_groups))

    E_best = None
    for pval in group_val.itervalues():
        if pval.value > E_best:
            E_best = pval.value
            best_paths = pval.paths

    print "Best path group found in {0}s, max E_kld = {1}".format(time.time()-t_arrange, E_best)
    return best_paths, E_best

p_lines = mc.LineCollection([])
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
        # vehicle.build_likelihood_tree(kld_depth)

        # Generate persistent probability map (for plotting)
        vehicle.belief.persistent_centre_probability_map()

        vehicle.setup_plot(hv, tree_depth=kld_depth)
        vehicle.update_plot()

    best_paths, E_best = arrange_meeting(roadmap, vehicles[0], vehicles[0].get_current_state(), max_cost=max_path_cost,
                                         n_robots=n_vehicles, min_nodes=2)
    print best_paths
    lseg, lcol = line_segments_from_paths(roadmap.G, best_paths)
    p_lines.set_segments(lseg)
    p_lines.set_color(lcol)
    vehicles[0].h_ax.add_collection(p_lines)

    for i, vehicle in enumerate(vehicles):
        vehicle.set_path(best_paths[i], E_best)

    all_artists = get_all_artists(vehicles)
    all_artists.append(p_lines)
    return all_artists


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

    # if dd[vm] > curr_dkl:
        # We share :)
    if all([vehicle.path_finished() for vehicle in vehicles]):
        last_share = belief_state.share_beliefs(vehicles, last_share)
        print "Shared at {0}".format(last_share)
        curr_dkl = dd[vm] + delta_dkl * max_dkl
        current_wait = 0
        best_paths, E_best = arrange_meeting(roadmap, vehicles[0], vehicles[0].get_current_state(),
                                             max_cost=max_path_cost,
                                             n_robots=n_vehicles, min_nodes=2)
        print best_paths
        lseg, lcol = line_segments_from_paths(roadmap.G, best_paths)
        p_lines.set_segments(lseg)
        p_lines.set_color(lcol)
        vehicles[0].h_ax.add_collection(p_lines)

        for i, vehicle in enumerate(vehicles):
            vehicle.set_path(best_paths[i], E_best)

        all_artists = get_all_artists(vehicles)
        all_artists.append(p_lines)
        return all_artists

    else:
        for vehicle in vehicles:
            vehicle.run_path()      # vehicle.kld_select_obs(kld_depth)
            vehicle.update_plot()

    for vehicle in vehicles:
        vehicle.belief.persistent_centre_probability_map()

    pcmax = max([v.h_artists['pc'].get_clim()[1] for v in vehicles])
    for v in vehicles:
        v.h_artists['pc'].set_clim([0, pcmax])

    print "i = {0}/{1}".format(i + 1, n_obs)

    all_artists = get_all_artists(vehicles)
    all_artists.append(p_lines)
    return all_artists


ani = animation.FuncAnimation(h_fig, animate, init_func=init, frames=n_obs, interval=100, blit=True, repeat=False)
# ani.save('../vid/temp.ogv', writer = 'avconv', fps=3, bitrate=5000, codec='libtheora')
h_fig.show()



       #return paths[best_path], E_kld[best_path]

    # For each group, sequentially greedily select the best path until have paths for all robots
    # # Now, for each group find the best set of observation locations
    # group_val = {}
    # for end_node, path_group in path_groups.iteritems():
    #     # For each path group, sequentially select the best location, then stop when no paths left
    #     nodes = set()
    #     for path in path_group.keys():
    #         nodes = nodes.union(set(path)) # Get unique nodes in the set
    #
    #     cpaths = []
    #     for node in nodes:
    #         cpaths.append((node,))
    #
    #     best_node, best_kld = vehicle.expected_kld_from_paths(cpaths)


# def max_E_kld_from_path_group(vehicle, path_group):
#
#     # For each path group, sequentially select the best location, then stop when no paths left
#     nodes = set()
#     for path in path_groups[end_node].keys():
#         nodes = nodes.union(set(path))  # Get unique nodes in the set
#
#     cpaths = []
#     for node in nodes:
#         cpaths.append((node,))
#
#     best_node, best_kld = vehicle.expected_kld_from_paths(cpaths)
#
#
# def available_paths(path_group, nodes, n_robots):
#     # Check if a path group has enough paths to cover the node set
#     valid_paths = 0
#     checked_paths = 0
#
#     while valid_paths < n_robots and checked_paths < len(path_group):
#         for path in path_group:
#             checked_paths += 1
#             for node in nodes:
#                 if node in path:
#                     valid_paths += 1
#                     break
#             if valid_paths >= n_robots:
#                 return True
#
#     return False

