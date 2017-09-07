import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import pdist, squareform
import collections
import matplotlib.cm as cm

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


class GraphMotion:
    def __init__(self, graph):
        self.G = graph

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


class GraphLikelihoodTreeNode(object):
    def __init__(self, state, motion_model, likelihood_function, inverse_depth=0):
        self.state = state
        self.motion_model = motion_model
        self.children = None
        self.likelihood_function = likelihood_function
        Flike = self.likelihood_function(self.motion_model.get_pose(self.state),
                                         False)  # Likelihood of observing False at this location across all possible centre locations (MC sampled)
        self.likelihood = [Flike, 1.0 - Flike]  # Full likelihood ([0] False, [1] True)
        self.add_children(inverse_depth)  # Go down the tree another layer
        self.node_colours = ['firebrick', 'green', 'cornflowerblue', 'orange', 'mediumorchid', 'lightseagreen']

    def add_children(self, inverse_depth):
        self.inverse_depth = inverse_depth
        if self.inverse_depth <= 0:
            return
        if self.children is None:
            end_states = self.motion_model.get_end_states(self.state)
            self.children = [GraphLikelihoodTreeNode(
                end_state,
                self.motion_model,
                self.likelihood_function,
                inverse_depth - 1) for end_state in end_states]
        else:
            for child in self.children:
                child.add_children(inverse_depth - 1)

    def kill_children(self):
        for child in self.children:
            child.kill_children()
        self.children = None

    def get_likelihood(self, index):
        return self.likelihood[index]

    def plot_tree(self, ax, colour_index=0):
        x0, y0 = self.state[0:2]
        ax.plot(x0, y0, 'o', color=self.node_colours[colour_index % len(self.node_colours)])
        if self.children is not None:
            for ii, child in enumerate(self.children):
                child.plot_tree(ax, colour_index + 1)
                tt = self.motion_model.get_trajectory(self.state, ii)
                ax.plot(tt[:, 0], tt[:, 1], '--', color='grey')

    def kld_path_utility(self, kld_function, decision_list, current_depth=0, curr_pzgc=[]):
        new_pzgc = copy.copy(curr_pzgc)
        new_pzgc.append(self.likelihood[0])
        kld_sum = 0.0
        if current_depth == 0:
            kld_sum = self.children[decision_list[0]].kld_path_utility(
                kld_function,
                decision_list,
                current_depth + 1,
                [])

        elif current_depth < len(decision_list):
            child_branch = decision_list[current_depth]
            for cl in self.likelihood:
                new_pzgc[-1] = cl
                kld_sum += self.children[child_branch].kld_path_utility(
                    kld_function,
                    decision_list,
                    current_depth + 1,
                    new_pzgc)
        else:
            for cl in self.likelihood:
                new_pzgc[-1] = cl
                kld_sum += kld_function(pzgc=np.array(new_pzgc))
                # print "kld_fun called! kld_sum={0:0.3f}".format(kld_sum)
        return kld_sum
