import numpy as np
from src import utils
import networkx as nx

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

  timer.toc(average=True, log_at=1, log_str='src.graph_utils.generate_graph')
  return (G)


