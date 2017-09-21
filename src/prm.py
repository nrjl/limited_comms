import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import pdist, squareform
import collections
import matplotlib.cm as cm
import matplotlib.collections as mc
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
        return "Graph with {0} nodes and {1} edges".format(self.get_number_vertices(), self.get_number_edges())

    def get_number_vertices(self):
        return len(self.V)

    def get_number_edges(self):
        return len(self.E)

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

    def get_all_locations(self):
        loc = np.zeros((self.get_number_vertices(), len(self.get_vertex(self.V.keys()[0]).location)), dtype='float')
        for i, v in enumerate(self.V):
            loc[i] = self.get_vertex(v).location
        return loc

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

    def no_edge_revisit_paths(self, current_path, current_budget=0.0, max_budget=()):
        # Find all paths (within budget) that traverse each (directed) edge once at most
        paths = []
        cV = self.V[current_path[-1]]
        budget_exceeded = True
        for v_id in cV.neighbours:
            new_edge = (cV.get_id(), v_id)
            if not edge_in_path(current_path, new_edge):
                ncost = float(current_budget + cV.neighbours[v_id])
                if ncost < max_budget:
                    budget_exceeded = False
                    paths.extend(self.no_edge_revisit_paths(current_path+(v_id,), ncost, max_budget))
        if budget_exceeded or len(paths) == 0:
            paths.append(current_path)
            # print "Path found: ", current_path
        return paths


def edge_in_path(path, edge):
    if len(path) >= 2:
        for i in range(len(path)-2):
            if edge[0] == path[i] and edge[1] == path[i+1]:
                return True
    return False

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
        self.distance_matrix = None
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

        self.calc_distance_matrix()

        for v_id, d in enumerate(self.distance_matrix):
            for u_id, dd in enumerate(d):
                if dd > 0 and dd < self.r:
                    self.G.add_edge(u_id, v_id, dd)
                    self.G.add_edge(v_id, u_id, dd)

    def calc_distance_matrix(self):
        # This might be dodgy - not guaranteed to pull vertices in order?
        loc = np.array([self.G.V[v].location for v in range(self.G.get_number_vertices())])
        self.distance_matrix = squareform(pdist(loc))              # Distance matrix

    def get_edge_cost_bounds(self):
        w_min, w_max = 0.0, 0.0
        for w in self.G.E.values():
            w_min, w_max = min(w, w_min), max(w, w_max)
        return w_min, w_max

    def plot_PRM(self, axh, label_nodes=False):
        v_locations = np.zeros((self.G.get_number_vertices(), 2))
        for i, v in enumerate(self.G.get_vertices()):
            v_locations[i] = v.get_location()
            if label_nodes:  axh.text(v_locations[i,0], v_locations[i,1], v.get_id())

        h_vertices = axh.plot(v_locations[:,0], v_locations[:,1], 'r.')

        w_min, w_max = self.get_edge_cost_bounds()
        cmap = cm.get_cmap()
        dw = w_max-w_min
        if dw <= 0:
            dw = 1.0

        Edone = set()

        lines = []
        colours = []

        for e_id, w in self.G.E.items():
            if e_id not in Edone and (e_id[1], e_id[0]) not in Edone:
                ec = (w-w_min)/dw
                p0 = self.G.get_vertex(e_id[0]).location
                p1 = self.G.get_vertex(e_id[1]).location
                lines.append([p0,p1])
                colours.append(cmap(ec))
                Edone.add(e_id)
        e_lines = mc.LineCollection(lines, colors=colours)
        axh.add_collection(e_lines)
        axh.set_xlim(self.limits[0])
        axh.set_ylim(self.limits[1])
        print "{0} vertices, {1} edges plotted".format(len(self.G.V), len(Edone))
        return h_vertices, e_lines

    def paths_to_cost(self, vertex_tuple, max_cost, min_nodes=0, ccost=0.0):
        assert type(vertex_tuple) is tuple, "Input must be a tuple"

        if ccost > max_cost:
            return {}

        else:
            paths = {}
            if len(vertex_tuple) >= min_nodes:
                paths[vertex_tuple] = ccost
            for n in self.G.get_neighbours(vertex_tuple[-1]):
                if n not in vertex_tuple:
                    paths.update(self.paths_to_cost(vertex_tuple + (n,), max_cost, min_nodes=min_nodes,
                                                       ccost=ccost+self.distance_matrix[vertex_tuple[-1], n]))
            return paths

    def pull_path_groups(self, start_vertex, max_cost, n_robots, min_nodes=0):
        if self.distance_matrix is None:
            self.calc_distance_matrix()
        paths = self.paths_to_cost((start_vertex,), max_cost, min_nodes)

        path_groups = {}
        for path in paths:
            if path[-1] not in path_groups:
                path_groups[path[-1]] = {path:paths[path]}
            else:
                # Check if new path is superset or subset of previous path
                paths_to_remove = []
                path_to_add = path
                pset = set(path[1:-1])

                for end_path in path_groups[path[-1]]:      #
                    if pset.issuperset(end_path[1:-1]):
                        paths_to_remove.append(end_path)
                    elif pset.issubset(end_path[1:-1]):
                        path_to_add = False
                        break

                if path_to_add:
                    path_groups[path[-1]].update({path_to_add:paths[path_to_add]})
                if paths_to_remove:
                    for pr in paths_to_remove:
                        del path_groups[path[-1]][pr]

        # Remove path groups with too few paths for the number of robots
        marked_for_deletion = []
        for end_node in path_groups:
            if len(path_groups[end_node]) < n_robots:
                marked_for_deletion.append(end_node)

        for node in marked_for_deletion:
            del path_groups[node]

        return path_groups



class GraphVehicle(belief_state.Vehicle):

    def _init_extras(self, *args, **kwargs):
        self.mc_likelihood = {}

    def _reset_extras(self, xs=None):
        self.mc_likelihood = {}
        if xs is not None:
            self.reset_mc_likelihood()

    def reset_mc_likelihood(self):
        self.mc_likelihood = {}
        ns = self.sensor.get_n_returns()
        for vtx in self.motion_model.get_graph().get_vertices():
            p_z_given_x = np.ones((ns, len(self.belief.csamples)), dtype='float')
            for z in range(ns-1):
                p_z_given_x[z] = [self.sensor.likelihood(vtx.get_location(), z, c=xc) for xc in self.belief.csamples]
                p_z_given_x[ns-1] -= p_z_given_x[z]     # Last row is 1.0-sum(p(z|x) forall z != ns
            self.mc_likelihood[vtx.get_id()] = p_z_given_x

    def get_mc_likelihood(self, v_id):
        # This returns the observation likelihood of all possible observations given a graph node (v_id) and Monte Carlo
        # sampled
        return self.mc_likelihood[v_id]

    def get_pose(self, v_id):
        return self.motion_model.G.get_vertex(v_id).get_location()

    def build_likelihood_tree(self,depth=3):
        # A likelihood tree stores the likelihood of observations over a tree of actions
        self.likelihood_tree = GraphLikelihoodTreeNode(
            self.get_current_state(),
            self.motion_model,
            self.get_mc_likelihood,
            inverse_depth=depth)

    def kld_tree(self, depth=3):
        # Deprecated ugly way to calculate KLD over path set because it requires LikelihoodTree
        # This gets all the paths, and each path *includes* the current node as the 0th node in the path
        all_paths = self.motion_model.get_tree_paths(self.get_current_state(), depth)
        end_nodes = {}
        self.k_tree = {}
        max_util = None

        for path in all_paths:
            # kld_path_utility assumes you're working from the current node (not included in path)
            # v_kld = self.likelihood_tree.kld_path_utility(self.belief.kld_likelihood, path[1:]) # DEPRECATED BECAUSE TREE IS DUMBFACE
            v_kld = self.expected_kld(path[1:], 0, self.belief.kld_likelihood, self.get_mc_likelihood)

            if path[-1] not in end_nodes or end_nodes[path[-1]] < v_kld:
                end_nodes[path[-1]] = v_kld
                self.k_tree[path] = v_kld
                if self.k_tree[path] > max_util:
                    max_util = self.k_tree[path]
                    best_path = path

        return self.k_tree, best_path, max_util

    def expected_kld(self, obs_locations, depth, kld_function, likelihood_function, pzgc=None):
        if pzgc is None:
            pzgc = np.ones((1, len(self.belief.csamples)), dtype='float')

        if depth >= len(obs_locations):
            return kld_function(pzgc)

        c_state = obs_locations[depth]
        E_kld = 0.0

        for cl in likelihood_function(c_state):
            # pzgc[depth] = cl
            E_kld += self.expected_kld(obs_locations, depth+1, kld_function, likelihood_function, pzgc*cl)

        return E_kld

    def expected_kld_from_paths(self, paths):
        E_kld = np.zeros(len(paths))
        for i, path in enumerate(paths):
            E_kld[i] = self.expected_kld(path, 0, self.belief.kld_likelihood, self.get_mc_likelihood)

        best_path = np.argmax(E_kld)

        return paths[best_path], E_kld[best_path]

    def kld_select_obs(self, depth):

        paths, best_path, max_util = self.kld_tree(depth=depth)

        next_state = best_path[1]
        # self.prune_likelihood_tree(next_state, depth)
        new_path = self.motion_model.get_trajectory(self.get_current_state(), next_state)
        self.full_path = np.append(self.full_path, new_path, axis=0)

        self.set_current_state(next_state)

        cobs = self.generate_observations([self.get_current_pose()])
        self.add_observations(cobs)

        return next_state

    def setup_tree_plot(self, tree_depth, ms_scatter):
        return self.h_ax.scatter([], [], cmap=cm.jet)

    def update_tree_plot(self):
        try:
            path_xy, ec = np.zeros((len(self.k_tree), 2)), np.zeros(len(self.k_tree))
            for i, path in enumerate(self.k_tree):
                path_xy[i] = self.get_pose(path[-1])
                ec[i] = self.k_tree[path]

            self.h_artists['tree'].set_offsets(path_xy)
            self.h_artists['tree'].set_array(ec - ec.min())

        except (KeyError, AttributeError):
            pass

    def set_path(self, path):
        self.path_index = 1 # Note that we do this because the first vertex is the current vertex
        self.meeting_path = path

    def run_path(self):
        if not self.path_finished():

            # Check if we should start a new path on our own
            next_state = self.meeting_path[self.path_index]
            new_path = self.motion_model.get_trajectory(self.get_current_state(), next_state)
            self.full_path = np.append(self.full_path, new_path, axis=0)

            self.set_current_state(next_state)

            cobs = self.generate_observations([self.get_current_pose()])
            self.add_observations(cobs)
            self.path_index += 1
        else:
            next_state = self.get_current_state()

        return next_state

    def path_finished(self):
        return self.path_index >= len(self.meeting_path)


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

    def get_tree_paths(self, v_id, depth=1):
        return self.G.acyclic_paths_to_depth((v_id,), depth)

    def get_paths_to_goal(self, start_id, goal_id, cost):
        return self.G.acyclic_paths_to_goal((start_id,), goal_id, max_cost = cost)

    def get_trajectory(self, u_id, v_id):
            #self.G.V[u_id].location + (self.G.V[v_id].location - self.G.V[v_id].location)*self.t
        return np.array([self.get_pose(u_id), self.get_pose(v_id)])


class GraphLikelihoodTreeNode(belief_state.LikelihoodTreeNode):
    def add_children(self, inverse_depth):
        self.inverse_depth = inverse_depth
        if self.inverse_depth <= 0:
            return
        if self.children is None:
            self.children={}
            end_states = self.motion_model.get_end_states(self.state)
            for end_state in end_states:
                self.children[end_state] = GraphLikelihoodTreeNode(end_state,
                self.motion_model,
                self.likelihood_function,
                inverse_depth - 1)
        else:
            for child in self.children.values():
                child.add_children(inverse_depth - 1)

    def plot_tree(self, ax, colour_index=0):
        x0, y0 = self.motion_model.get_pose(self.state)[0:2]
        ax.plot(x0, y0, 'o', color=self.node_colours[colour_index % len(self.node_colours)])
        if self.children is not None:
            for ii, child in enumerate(self.children):
                child.plot_tree(ax, colour_index + 1)
                tt = self.motion_model.get_trajectory(self.state, child.state)
                ax.plot(tt[:, 0], tt[:, 1], '--', color='grey')


