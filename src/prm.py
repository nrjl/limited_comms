import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import pdist, squareform
import collections
import matplotlib.cm as cm
import belief_state

class Edge(object):
    def __init__(self, u_id, v_id, weight):
        self.id = (u_id, v_id)
        self.weight = weight

    def __hash__(self):
        return self.id.__hash__()

class Graph(object):
    def __init__(self):
        self.V = {}
        self.E = {}
        # self.add_vertices(V)

    def __iter__(self):
        return iter(self.V.values())

    def __str__(self):
        return "Graph with {0} nodes and {1} edges".format(len(self.V), len(self.E))

    def print_vertices(self):
        for v in self.V.values():
            print v

    def add_vertex(self, v):
        self.V[v.get_id()] = v

    def add_vertices(self, newV):
        for v in newV:
            self.add_vertex(v)

    def get_vertex(self, id):
        return self.V[id]

    def add_edge(self, u_id, v_id, weight = 0):
        assert u_id in self.V and v_id in self.V, "Vertex ID not found, vertices must be in graph before edge can be added"
        self.V[u_id].add_neighbor(self.V[v_id], weight)
        self.E[(u_id, v_id)] = weight

    def get_vertices(self):
        return self.V.values()

    def get_neighbours(self, v_id):
        return self.V[v_id].neighbours.keys()

    def get_edge_weight(self, u_id, v_id):
        return self.V[u_id].neighbours[v_id]

    def acyclic_paths_to_depth(self, vertex_tuple, depth=3):
        assert type(vertex_tuple) is tuple, "Input must be a tuple"

        if len(vertex_tuple) == depth+1:
            return [vertex_tuple]

        else:
            paths = []
            for n in self.get_neighbours(vertex_tuple[-1]):
                if n not in vertex_tuple:
                    paths.extend(self.acyclic_paths_to_depth(vertex_tuple+(n,), depth))
            return paths


    def acyclic_paths_to_goal(self, vertex_tuple, goal_vertex, ccost = 0.0, max_cost=()):
        # max_cost defaults to () > anything always returns true (maybe not valid in Python 3.0?)
        assert type(vertex_tuple) is tuple, "Input must be a tuple"

        if vertex_tuple[-1] is goal_vertex and ccost < max_cost:
            return {vertex_tuple:ccost}

        elif ccost < max_cost:
            paths = {}
            for n in self.get_neighbours(vertex_tuple[-1]):
                if n not in vertex_tuple:
                    dcost = ccost + self.get_edge_weight(vertex_tuple[-1], n)
                    paths.update(self.acyclic_paths_to_goal(vertex_tuple+(n,), goal_vertex, dcost, max_cost))
            return paths

        else:
            return {}


class Vertex(object):
    def __init__(self, id, location):
        assert isinstance(id, collections.Hashable), "Node ID must be hashable"
        self.id = id
        self.location = location
        self.neighbours = {}

    def __str__(self):
        return "Vertex {0} at {1}, neighbours: {2}".format(self.id, self.location, self.neighbours.keys())

    def __hash__(self):
        return self.id.__hash__()

    def get_id(self):
        return self.id

    def get_location(self):
        return self.location

    def add_neighbor(self, v, weight=0):
        self.neighbours[v.get_id()] = weight


class PRM(object):

    def __init__(self, limits, n_points, **kwargs):
        self.limits = np.array(limits)
        assert len(self.limits.shape) == 2, 'Limits must be a d x 2 array'
        assert self.limits.shape[1] == 2, 'Limits must be a d x 2 array'
        assert np.all(self.limits[:,1] - self.limits[:,0]) > 0, 'Limits must be increasing'
        self.d = self.limits.shape[0]
        self.n_points = n_points
        self.G = Graph()
        self.build_PRM(**kwargs)

    def __str__(self):
        return "PRM: {0}-dimensional, type: {1}, n={2}, k={3}, r={4}".format(self.d, self.type, self.n_points, self.k, self.r)

    def get_vertex(self, u_id):
        return self.G.get_vertex(u_id)

    def build_PRM(self, type='kPRM*', k=None, r=None):
        self.type = type
        self.k = k
        self.r = r

        if type == 'kPRM*':
            self.k = int(np.ceil(np.exp(1)*(1 + 1.0/self.d)))
        elif type == 'kPRM':
            assert self.k is not None, 'k not specified, must specify a k value for kPRM'
        elif type == 'PRM*':
            # Lebesgue measure of space (ignoring obstacles)
            space_vol = np.diff(self.limits).prod()
            # Volume of unit ball
            V_ball = np.pi**(self.d/2.0)/gamma(self.d/2.0 + 1)
            d_inv = 1.0/self.d
            gamma_star = 2.0*((1 + d_inv)*(space_vol/V_ball))**(d_inv)
            self.r = gamma_star*(np.log(self.n_points)/self.n_points)**(d_inv)
            self.k = None
        elif type == 'sPRM':
            assert self.r is not None, 'r not specified, must specify an r value for sPRM'
            self.k = None
        else:
            raise ValueError('build_PRM type specification: {0} not recognized'.format(type))

        if self.k is not None:
            self.build_kPRM()
        else:
            self.build_sPRM()

    def sample_nodes(self):
        # Place vertices (no obstacles at the moment)
        locations = np.random.rand(self.n_points, self.d)
        locations = locations*np.diff(self.limits).T + self.limits[:,0].T
        V = [Vertex(i, loc) for i, loc in enumerate(locations)]
        return V

    def build_kPRM(self):
        V = self.sample_nodes()
        self.G.add_vertices(V)

        loc = np.array([v.location for v in V])
        D = squareform(pdist(loc))              # Distance matrix

        for v_id, d in enumerate(D):
            U = d.argsort()[:(self.k+1)]  # This gets the k+1 smallest indices (since distance to self is 0)
            for u_id in U:
                if u_id != v_id:
                    self.G.add_edge(u_id, v_id, d[u_id])
                    self.G.add_edge(v_id, u_id, d[u_id])

    def build_sPRM(self):
        V = self.sample_nodes()
        self.G.add_vertices(V)

        loc = np.array([v.location for v in V])
        D = squareform(pdist(loc))              # Distance matrix

        for v_id, d in enumerate(D):
            for u_id, dd in enumerate(d):
                if dd > 0 and dd < self.r:
                    self.G.add_edge(u_id, v_id, dd)
                    self.G.add_edge(v_id, u_id, dd)

    def get_edge_cost_bounds(self):
        w_min, w_max = 0.0, 0.0
        for w in self.G.E.values():
            w_min, w_max = min(w, w_min), max(w, w_max)
        return w_min, w_max

    def plot_PRM(self, axh, label_nodes=False):
        for v in self.G.get_vertices():
            axh.plot(v.location[0], v.location[1], 'r.')
            if label_nodes: axh.text(v.location[0], v.location[1], v.get_id())

        w_min, w_max = self.get_edge_cost_bounds()
        cmap = cm.get_cmap()
        dw = w_max-w_min
        if dw <= 0: dw=1.0

        Edone = set()

        for e_id, w in self.G.E.items():
            if e_id not in Edone and (e_id[1], e_id[0]) not in Edone:
                ec = (w-w_min)/dw
                x0, y0 = self.G.get_vertex(e_id[0]).location
                x1, y1 = self.G.get_vertex(e_id[1]).location
                axh.plot([x0, x1], [y0, y1], color=cmap(ec))
                Edone.add(e_id)

        axh.set_xlim(self.limits[0])
        axh.set_ylim(self.limits[1])
        print "{0} vertices, {1} edges plotted".format(len(self.G.V), len(Edone))
        return w_min, w_max


class GraphVehicle(belief_state.Vehicle):

    def _init_extras(self, *args, **kwargs):
        self.mc_likelihood = {}

    def _reset_extras(self):
        self.mc_likelihood = {}

    def reset_mc_likelihood(self):
        self.mc_likelihood = {}

        for vtx in self.motion_model.get_graph().get_vertices():
            p_F_given_x = np.array([self.obs_fun(vtx.get_location(), False, c=xc) for xc in self.belief.csamples])
            self.mc_likelihood[vtx.get_id()] = np.vstack((p_F_given_x, 1.0 - p_F_given_x))

    def get_mc_likelihood(self, v_id, z):
            return self.mc_likelihood[v_id][z]

    def get_pose(self, v_id):
        return self.motion_model.G.get_vertex(v_id).get_location()

    def build_likelihood_tree(self,depth=3):
        # A likelihood tree stores the likelihood of observations over a tree of actions
        self.likelihood_tree = belief_state.LikelihoodTreeNode(
            self.get_current_state(),
            self.motion_model,
            self.get_mc_likelihood,
            inverse_depth=depth)

    # def kld_tree(self, vertex_tuple=None, depth=3):
    #     if vertex_tuple is None:
    #         vertex_tuple = (self.get_current_state(),)
    #     elif len(vertex_tuple) >= depth+1:
    #         return {vertex_tuple: self.likelihood_tree.kld_path_utility(self.belief.kld_likelihood, vertex_tuple)}
    #
    #     k_tree = {}
    #
    #     for v_id in self.motion_model.G.V[vertex_tuple].neighbours:
    #         if v_id is not vertex_tuple[-1]:
    #             k_tree.update(self.kld_tree(vertex_tuple+v_id, depth))
    #
    #     return k_tree
    #
    # def kld_select_obs(self, depth):
    #     max_util = None
    #     paths = self.kld_tree(depth)
    #     for path, path_util in paths:
    #         if path_util > max_util:
    #             best_path = path
    #             max_util = path_util
    #
    #     self.leaf_states = self.motion_model.get_leaf_states(self.get_current_state(), depth)
    #     self.next_states = self.motion_model.get_leaf_states(self.get_current_state(), 1)
    #
    #     self.leaf_values = np.array(self.kld_tree(depth))
    #     path_max = np.unravel_index(np.argmax(self.leaf_values),
    #                                 self.motion_model.get_paths_number() * np.ones(depth, dtype='int'))
    #     amax = path_max[0]
    #
    #     self.prune_likelihood_tree(amax, depth)
    #     self.full_path = np.append(self.full_path, self.motion_model.get_trajectory(self.get_current_pose(), amax),
    #                                axis=0)
    #
    #     self.set_current_state(self.next_states[amax])
    #
    #     cobs = self.generate_observations([self.get_current_pose()[0:2]])
    #     self.add_observations(cobs)
    #
    #     return amax
    #
    # def prune_likelihood_tree(self, selected_option, depth):
    #     self.likelihood_tree.children[selected_option].add_children(depth)
    #     self.likelihood_tree = self.likelihood_tree.children[selected_option]
    #
    # def setup_plot(self, h_ax, tree_depth=None, obs_symbols=['r^', 'go'], ms_start=8, ms_target=10, ms_scatter=20,
    #                ms_obs=6.5):
    #     self._plots = True
    #     h_ax.clear()
    #     self.h_ax = h_ax
    #     self.h_artists = {}
    #     self.h_artists['pc'] = self.h_ax.imshow(np.zeros(self.world.get_size()), origin='lower', vmin=0, animated=True)
    #     self.h_artists['cpos'], = self.h_ax.plot([], [], 'o', color='gold', fillstyle='full', ms=ms_start, mew=0)
    #     target_pos = self.world.get_target_location()
    #     self.h_artists['target'], = self.h_ax.plot(target_pos[0], target_pos[1], 'wx', mew=2, ms=ms_target)
    #     self.h_artists['start'], = self.h_ax.plot(self.start_pose[0], self.start_pose[1], '^', color='orange',
    #                                               ms=ms_start, fillstyle='full')
    #     self.h_artists['obsF'], = self.h_ax.plot([], [], obs_symbols[0], mew=0.5, mec='w', ms=ms_obs)
    #     self.h_artists['obsT'], = self.h_ax.plot([], [], obs_symbols[1], mew=0.5, mec='w', ms=ms_obs)
    #     self.h_artists['path'], = self.h_ax.plot([], [], 'w-', lw=2)
    #     self.h_ax.set_xlim(-.5, self.world.get_size()[0] - 0.5)
    #     self.h_ax.set_ylim(-.5, self.world.get_size()[1] - 0.5)
    #     if tree_depth is not None:
    #         self.leaf_poses = self.motion_model.get_leaf_poses(self.start_pose, depth=tree_depth)
    #         self.h_artists['tree'] = self.h_ax.scatter(self.leaf_poses[:, 0], self.leaf_poses[:, 1], ms_scatter)
    #
    #     if self.unshared:
    #         self.h_artists['shared_obsF'], = self.h_ax.plot([], [], '^', color='darksalmon', mec='w', mew=0,
    #                                                         ms=ms_obs - 1.5)
    #         self.h_artists['shared_obsT'], = self.h_ax.plot([], [], 'o', color='darkseagreen', mec='w', mew=0,
    #                                                         ms=ms_obs - 1.5)
    #
    # def update_plot(self):
    #     cpos = self.get_current_pose()
    #     self.h_artists['cpos'].set_data(cpos[0], cpos[1])
    #
    #     if self.belief.update_pc_map:
    #         pc = self.belief.persistent_centre_probability_map()
    #         pc = pc / pc.sum()
    #         self.h_artists['pc'].set_data(pc.transpose())
    #         self.h_artists['pc'].set_clim([0, pc.max()])
    #
    #     obsT = [xx for xx, zz in self.belief.get_observations() if zz == True]
    #     obsF = [xx for xx, zz in self.belief.get_observations() if zz == False]
    #     self.update_obs(self.h_artists['obsT'], obsT)
    #     self.update_obs(self.h_artists['obsF'], obsF)
    #
    #     self.h_artists['path'].set_data(self.full_path[:, 0], self.full_path[:, 1])
    #
    #     try:
    #         self.h_artists['tree'].set_offsets(self.leaf_poses[:, 0:2])
    #         self.h_artists['tree'].set_array(self.leaf_values - self.leaf_values.min())
    #     except (KeyError, AttributeError):
    #         pass
    #
    #         # return self.h_artists.values()
    #
    # def get_artists(self):
    #     # This is because stupid animate doesn't repsect plot order, so I can't just return h_artsists.values()
    #     if self.unshared:
    #         return (self.h_artists['pc'], self.h_artists['cpos'], self.h_artists['target'],
    #                 self.h_artists['start'], self.h_artists['obsT'], self.h_artists['obsF'],
    #                 self.h_artists['path'], self.h_artists['tree'],
    #                 self.h_artists['shared_obsT'], self.h_artists['shared_obsF'])
    #     else:
    #         return (self.h_artists['pc'], self.h_artists['cpos'], self.h_artists['target'],
    #                 self.h_artists['start'], self.h_artists['obsT'], self.h_artists['obsF'],
    #                 self.h_artists['path'], self.h_artists['tree'])
    #
    # def update_obs(self, h, obs):
    #     if obs != []:
    #         h.set_data(*zip(*obs))
    #
    # def add_observations(self, obs, *args, **kwargs):
    #     self.belief.add_observations(obs, *args, **kwargs)
    #
    # def get_observations(self):
    #     return self.belief.get_observations()
    #
    # def reset_observations(self):
    #     self.belief.reset_observations()


class GraphMotion:
    def __init__(self, graph):
        self.G = graph

    def get_graph(self):
        return self.G

    def get_vertex(self, u_id):
        return self.G.get_vertex(u_id)

    def get_pose(self, u_id):
        return self.get_vertex(u_id).get_location()

    def get_paths_number(self, v_id):
        return len(self.G.V[v_id].neighbours)

    def get_end_states(self, v_id):
        return self.G.V[v_id].neighbours.keys()

    def get_leaf_states(self, v_id, depth=1):
        return self.G.acyclic_paths_to_depth((v_id,), depth)
        # op = self.get_end_states(v_id)
        # if depth > 1:
        #     rp = []
        #     op = self.get_end_states(v_id)
        #     for state in op:
        #         rp.extend(self.get_leaf_states(state, depth - 1))
        # else:
        #     rp = op
        # return rp

    def get_paths_to_goal(self, start_id, goal_id, cost):
        return self.G.acyclic_paths_to_depth((start_id,), goal_id, max_cost = cost)

    def get_trajectory(self, u_id, v_id):
        poses = self.G.V[u_id].location + (self.G.V[v_id].location - self.G.V[v_id].location)*self.t
        return poses
