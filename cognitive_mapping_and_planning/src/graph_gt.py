"""Graph class to abstract out the graph library being used.
"""
import skimage.morphology
import numpy as np
import networkx as nx
import itertools
import graph_tool as gt
import graph_tool.topology
import graph_tool.generation 
import src.utils as utils

class Graph():
  def __init__(self, nxG):
    self.gtG, self.nodes_array, nodes_to_id = \
      self.convert_to_graph_tool(nxG)

  def convert_to_graph_tool(self, nxG):
    timer = utils.Timer()
    timer.tic()
    gtG = gt.Graph(directed=nxG.is_directed())
    gtG.ep['action'] = gtG.new_edge_property('int')

    nodes_list = nxG.nodes()
    nodes_array = np.array(nodes_list)

    nodes_id = np.zeros((nodes_array.shape[0],), dtype=np.int64)

    for i in range(nodes_array.shape[0]):
      v = gtG.add_vertex()
      nodes_id[i] = int(v)

    # d = {key: value for (key, value) in zip(nodes_list, nodes_id)}
    d = dict(itertools.izip(nodes_list, nodes_id))

    for src, dst, data in nxG.edges_iter(data=True):
      e = gtG.add_edge(d[src], d[dst])
      gtG.ep['action'][e] = data['action']
    nodes_to_id = d
    timer.toc(average=True, log_at=1, log_str='src.graph_utils.convert_to_graph_tool')
    return gtG, nodes_array, nodes_to_id
  
  def shortest_distance(self, source, target, weights=None, 
    reversed=False, pred_map=False, max_dist=None):
    if pred_map:
      dist, pred = gt.topology.shortest_distance(gt.GraphView(self.gtG, reversed=reversed),
        source=self.gtG.vertex(int(source)), target=target, weights=weights
        max_dist=max_dist, pred_map=pred_map)
      dist = np.array(dist.get_array())
      pred = np.array(pred.get_array())
      return dist, pred
    else:
      dist = gt.topology.shortest_distance(gt.GraphView(self.gtG, reversed=reversed),
        source=self.gtG.vertex(int(source)), target=target, weights=weights
        max_dist=max_dist, pred_map=pred_map)
      dist = np.array(dist.get_array())
      pred = np.array(pred.get_array())
      return dist, pred

  # Compute shortest distance from all nodes to or from the set of all source
  # nodes.
  def get_distance_node_list(self, source_nodes, direction, weights=None):
    gtG_ = gt.Graph(self.gtG)
    v = gtG_.add_vertex()
    
    if weights is not None:
      weights = gtG_.edge_properties[weights]
    
    for s in source_nodes:
      e = gtG_.add_edge(s, int(v))
      if weights is not None:
        weights[e] = 0.

    if direction == 'to':
      dist = gt.topology.shortest_distance(
          gt.GraphView(gtG_, reversed=True), source=gtG_.vertex(int(v)),
          target=None, weights=weights)
    elif direction == 'from':
      dist = gt.topology.shortest_distance(
          gt.GraphView(gtG_, reversed=False), source=gtG_.vertex(int(v)),
          target=None, weights=weights)
    dist = np.array(dist.get_array())
    dist = dist[:-1]
    if weights is None:
      dist = dist-1
    return dist

# Functions for semantically labelling nodes in the traversal graph.
def _generate_lattice(sz_x, sz_y):
  """Generates a lattice with sz_x vertices along x and sz_y vertices along y
  direction Each of these vertices is step_size distance apart. Origin is at
  (0,0).  """
  g = gt.generation.lattice([sz_x, sz_y])
  x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
  x = np.reshape(x, [-1,1]); y = np.reshape(y, [-1,1]);
  nodes = np.concatenate((x,y), axis=1)
  return g, nodes

def _add_diagonal_edges(g, nodes, sz_x, sz_y, edge_len):
  offset = [sz_x+1, sz_x-1]
  for o in offset:
    s = np.arange(nodes.shape[0]-o-1)
    t = s + o
    ind = np.all(np.abs(nodes[s,:] - nodes[t,:]) == np.array([[1,1]]), axis=1)
    s = s[ind][:,np.newaxis]
    t = t[ind][:,np.newaxis]
    st = np.concatenate((s,t), axis=1)
    for i in range(st.shape[0]):
      e = g.add_edge(st[i,0], st[i,1], add_missing=False)
      g.ep['wts'][e] = edge_len

def _convert_traversible_to_graph(traversible, ff_cost=1., fo_cost=1.,
                                 oo_cost=1., connectivity=4):
  assert(connectivity == 4 or connectivity == 8)

  sz_x = traversible.shape[1]
  sz_y = traversible.shape[0]
  g, nodes = _generate_lattice(sz_x, sz_y)

  # Assign costs.
  edge_wts = g.new_edge_property('float')
  g.edge_properties['wts'] = edge_wts
  wts = np.ones(g.num_edges(), dtype=np.float32)
  edge_wts.get_array()[:] = wts

  if connectivity == 8:
    _add_diagonal_edges(g, nodes, sz_x, sz_y, np.sqrt(2.))

  se = np.array([[int(e.source()), int(e.target())] for e in g.edges()])
  s_xy = nodes[se[:,0]]
  t_xy = nodes[se[:,1]]
  s_t = np.ravel_multi_index((s_xy[:,1], s_xy[:,0]), traversible.shape)
  t_t = np.ravel_multi_index((t_xy[:,1], t_xy[:,0]), traversible.shape)
  s_t = traversible.ravel()[s_t]
  t_t = traversible.ravel()[t_t]

  wts = np.zeros(g.num_edges(), dtype=np.float32)
  wts[np.logical_and(s_t == True, t_t == True)] = ff_cost
  wts[np.logical_and(s_t == False, t_t == False)] = oo_cost
  wts[np.logical_xor(s_t, t_t)] = fo_cost

  edge_wts = g.edge_properties['wts']
  for i, e in enumerate(g.edges()):
    edge_wts[e] = edge_wts[e] * wts[i]
  # d = edge_wts.get_array()*1.
  # edge_wts.get_array()[:] = d*wts 
  return g, nodes

def label_nodes_with_class_geodesic(nodes_xyt, class_maps, pix, traversible,
    ff_cost=1., fo_cost=1., oo_cost=1., connectivity=4):
  """Labels nodes in nodes_xyt with class labels using geodesic distance as
  defined by traversible from class_maps.
  Inputs:
    nodes_xyt
    class_maps: counts for each class.
    pix: distance threshold to consider close enough to target.
    traversible: binary map of whether traversible or not.
  Output:
    labels: For each node in nodes_xyt returns a label of the class or -1 is
    unlabelled.
  """
  g, nodes = _convert_traversible_to_graph(traversible, ff_cost=ff_cost,
    fo_cost=fo_cost, oo_cost=oo_cost, connectivity=connectivity)

  class_dist = np.zeros_like(class_maps*1.)
  n_classes = class_maps.shape[2]
  if False:
    # Assign each pixel to a class based on number of points.
    selem = skimage.morphology.disk(pix)
    class_maps_ = class_maps*1.
    class_maps__ = np.argmax(class_maps_, axis=2)
    class_maps__[np.max(class_maps_, axis=2) == 0] = -1

  # Label nodes with classes.
  for i in range(n_classes):
    # class_node_ids = np.where(class_maps__.ravel() == i)[0]
    class_node_ids = np.where(class_maps[:,:,i].ravel() > 0)[0]
    dist_i = g.get_distance_node_list(class_node_ids, 'to', weights='wts')
    class_dist[:,:,i] = np.reshape(dist_i, class_dist[:,:,i].shape)
  class_map_geodesic = (class_dist <= pix)
  class_map_geodesic = np.reshape(class_map_geodesic, [-1, n_classes])

  # For each node pick out the label from this class map.
  x = np.round(nodes_xyt[:,[0]]).astype(np.int32)
  y = np.round(nodes_xyt[:,[1]]).astype(np.int32)
  ind = np.ravel_multi_index((y,x), class_dist[:,:,0].shape)
  node_class_label = class_map_geodesic[ind[:,0],:]
  class_map_geodesic = class_dist <= pix
  return class_map_geodesic, node_class_label



