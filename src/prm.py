import numpy as np
from scipy.special import gamma
from scipy.spatial.distance import pdist
import collections

class Graph(object):

    def __init__(self, V, E):
        self.V = V
        self.E = E


class UndirectedGraph(Graph):

    def build_connectivity(self):
        connectivity = np.array((len(self.V), len(self.V)), dtype='bool')


class Edge(object):

    def __init__(self, V0, V1, cost):
        self.V0 = V0
        self.V1 = V1
        self.cost = cost

class Graph(object):
    def __init__(self, V=[]):
        self.V = {}
        self.add_vertices(V)

    def __iter__(self):
        return iter(self.V.values())

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

    def get_vertices(self):
        return self.V.values()

class Vertex(object):
    def __init__(self, id, location):
        assert isinstance(id, collections.Hashable), "Node ID must be hashable"
        self.id = id
        self.location = location
        self.neighbours = {}

    def __hash__(self):
        return self.id.__hash__()

    def get_id(self):
        return self.id

    def add_neighbor(self, v, weight=0):
        self.neighbours[v.get_id()] = weight


class PRM(object):

    def __init__(self, limits, n_points, type='kPRM*', k=None, r=None):
        self.limits = np.array(limits)
        assert len(self.limits.shape) == 2, 'Limits must be a d x 2 array'
        assert self.limits.shape[1] == 2, 'Limits must be a d x 2 array'
        assert np.all(self.limits[:,1] - self.limits[:,0]) > 0, 'Limits must be increasing'
        self.d = limits.shape[0]
        self.n_points = n_points
        self.k = k
        self.r = r
        self.build_PRM(type)

    def build_PRM(self, type):
        self.type = type

        if type == 'kPRM*':
            self.k = np.exp(1)*(1 + 1.0/self.d)

        elif type == 'kPRM':
            assert self.k is not None, 'k not specified, must specify a k value for kPRM'
        elif type == 'PRM*':
            # Lebesgue measure of space (ignoring obstacles)
            space_vol = np.diff(self.limits).prod()
            # Volume of unit ball
            V_ball = np.pi**(self.d/2.0)/gamma(self.d/2.0 + 1)
            self.r = 2.0*(1 + 1.0/self.d)**(1.0/self.d)*(space_vol/V_ball)**(1.0/self.d)
            self.k = None
        elif type == 'sPRM':
            assert self.r is not None, 'r not specified, must specify an r value for sPRM'
            self.k = None
        else:
            raise ValueError('build_PRM type specification: %0 not recognized'.format(type))

        if self.k is None:
            self.build_kPRM()
        else:
            self.build_rPRM()

    def sample_nodes(self):
        # Place vertices (no obstacles at the moment)
        locations = np.random.rand(self.n_points, self.d)
        self.V = [Vertex(loc) for loc in locations]
        return locations

    def build_kPRM(self):
        loc = self.sample_nodes()   # Sample all locations
        D = pdist(loc)              # Distance matrix
        self.E = set()

        for v, d in enumerate(D):
            U = d.argsort()[:(self.k+1)]  # This gets the k+1 smallest indices (since distance to self is 0)
            for u in U:
                if u != v:
                    self.E.add(Edge(u, v, d[u]))
                    self.E.add(Edge(v, u, d[u]))







