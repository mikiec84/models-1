import numpy as np
import itertools
from src import utils
import networkx as nx

def label_nodes_with_class_geodesic(nodes_xyt, class_maps, pix, traversible,
    ff_cost=1., fo_cost=1., oo_cost=1., connectivity=4):
  raise NotImplemented

def generate_graph(valid_fn_vec=None, sc=1., n_ori=6,
  starting_location=(0, 0, 0), vis=False, directed=True):
  
  nxG = generate_graph_helper(valid_fn_vec=valid_fn_vec, sc=sc, 
    n_ori=n_ori, starting_location=starting_location, vis=vis, 
    directed=directed)
  nodes_list = nxG.nodes()
  nodes_array = np.array(nodes_list)
  nodes_id = np.zeros((nodes_array.shape[0],), dtype=np.int64)
  for i in range(nodes_array.shape[0]):
    nodes_id[i] = i 
  d = dict(itertools.izip(nodes_list, nodes_id))
  nodes_to_id = d
  nxG_ = nx.relabel_nodes(nxG, mapping=d)
  return Graph(nxG_), nodes_array, nodes_to_id

class Graph():
  def __init__(self, nxG):
    self.nxG = nxG
    self.num_vertex = len(self.nxG.nodes())
    self.reversed_nxG = None

  def to_array(self, d, num_vertex, init_val):
    out = init_val * np.ones((num_vertex, ), dtype=type(init_val))
    for k, v in d.items():
      out[k] = v
    return out
  
  def shortest_distance(self, source, target, weights=None, reversed=False, 
    pred_map=False, max_dist=None):
    assert(weights is None), 'Only supports unweighted graphs right now.'
    if reversed:
      if self.reversed_nxG is None:
        self.reversed_nxG = self.nxG.reverse()
      g = self.reversed_nxG
    else:
      g = self.nxG
    assert(target is None)
    assert(not pred_map)
    pred, dist = nx.dijkstra_predecessor_and_distance(g, source, cutoff=max_dist)
    dist = self.to_array(dist, self.num_vertex, np.int32(np.iinfo(np.int32).max))
    return dist
  
  def get_distance_node_list(self, source_nodes, direction, weights=None):
    raise NotImplemented

  def num_edges(self):
    return len(self.nxG.edges())

  def num_vertices(self):
    return len(self.nxG.nodes())
  
  def get_neighbours(self, c):
    neigh_edge = self.nxG.edges(data='action', nbunch=[c])
    _, neigh, neigh_action = zip(*neigh_edge)
    return neigh, neigh_edge, neigh_action 

def generate_graph_helper(valid_fn_vec=None, sc=1., n_ori=6,
  starting_location=(0, 0, 0), vis=False, directed=True):
  timer = utils.Timer()
  timer.tic()
  if directed: G = nx.DiGraph(directed=True)
  else: G = nx.Graph()
  G.add_node(starting_location)
  new_nodes = G.nodes()
  while len(new_nodes) != 0:
    nodes_to_add = []
    nodes_to_validate = []
    for n in new_nodes:
      if directed:
        na, nv = _get_next_nodes(n, sc, n_ori)
      else:
        na, nv = _get_next_nodes_undirected(n, sc, n_ori)
      nodes_to_add = nodes_to_add + na
      if valid_fn_vec is not None:
        nodes_to_validate = nodes_to_validate + nv
      else:
        node_to_add = nodes_to_add + nv

    # Validate nodes.
    vs = [_[1] for _ in nodes_to_validate]
    valids = valid_fn_vec(vs)

    for nva, valid in zip(nodes_to_validate, valids):
      if valid:
        nodes_to_add.append(nva)

    new_nodes = []
    for n,v,a in nodes_to_add:
      if not G.has_node(v):
        new_nodes.append(v)
      G.add_edge(n, v, action=a)

  timer.toc(average=True, log_at=1, log_str='src.graph_nx.generate_graph_helper')
  return (G)

def _get_next_nodes_undirected(n, sc, n_ori):
  nodes_to_add = []
  nodes_to_validate = []
  (p, q, r) = n
  nodes_to_add.append((n, (p, q, r), 0))
  if n_ori == 4:
    for _ in [1, 2, 3, 4]:
      if _ == 1:
        v = (p - sc, q, r)
      elif _ == 2:
        v = (p + sc, q, r)
      elif _ == 3:
        v = (p, q - sc, r)
      elif _ == 4:
        v = (p, q + sc, r)
      nodes_to_validate.append((n, v, _))
  return nodes_to_add, nodes_to_validate

def _get_next_nodes(n, sc, n_ori):
  nodes_to_add = []
  nodes_to_validate = []
  (p, q, r) = n
  for r_, a_ in zip([-1, 0, 1], [1, 0, 2]):
    nodes_to_add.append((n, (p, q, np.mod(r+r_, n_ori)), a_))

  if n_ori == 6:
    if r == 0:
      v = (p + sc, q, r)
    elif r == 1:
      v = (p + sc, q + sc, r)
    elif r == 2:
      v = (p, q + sc, r)
    elif r == 3:
      v = (p - sc, q, r)
    elif r == 4:
      v = (p - sc, q - sc, r)
    elif r == 5:
      v = (p, q - sc, r)
  elif n_ori == 4:
    if r == 0:
      v = (p + sc, q, r)
    elif r == 1:
      v = (p, q + sc, r)
    elif r == 2:
      v = (p - sc, q, r)
    elif r == 3:
      v = (p, q - sc, r)
  nodes_to_validate.append((n,v,3))

  return nodes_to_add, nodes_to_validate


