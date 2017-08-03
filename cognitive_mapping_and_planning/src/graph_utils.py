# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Various function to manipulate graphs for computing distances.
"""
import skimage.morphology
import numpy as np
import networkx as nx
import itertools
import src.utils as utils

import src.graph_gt as graph_lib

Graph = graph_lib.Graph
label_nodes_with_class_geodesic = graph_lib.Graph

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

def generate_graph(valid_fn_vec=None, sc=1., n_ori=6,
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

def vis_G(G, ax, vertex_color='r', edge_color='b', r=None):
  if edge_color is not None:
    for e in G.edges():
      XYT = zip(*e)
      x = XYT[-3]
      y = XYT[-2]
      t = XYT[-1]
      if r is None or t[0] == r:
        ax.plot(x, y, edge_color)
  if vertex_color is not None:
    XYT = zip(*G.nodes())
    x = XYT[-3]
    y = XYT[-2]
    t = XYT[-1]
    ax.plot(x, y, vertex_color + '.')

def _rejection_sampling(rng, sampling_d, target_d, bins, hardness, M):
  bin_ind = np.digitize(hardness, bins)-1
  i = 0
  ratio = target_d[bin_ind] / (M*sampling_d[bin_ind])
  while i < ratio.size and rng.rand() > ratio[i]:
    i = i+1
  return i

def heuristic_fn_vec(n1, n2, n_ori, step_size):
  # n1 is a vector and n2 is a single point.
  dx = (n1[:,0] - n2[0,0])/step_size
  dy = (n1[:,1] - n2[0,1])/step_size
  dt = n1[:,2] - n2[0,2]
  dt = np.mod(dt, n_ori)
  dt = np.minimum(dt, n_ori-dt)

  if n_ori == 6:
    if dx*dy > 0:
      d = np.maximum(np.abs(dx), np.abs(dy))
    else:
      d = np.abs(dy-dx)
  elif n_ori == 4:
    d = np.abs(dx) + np.abs(dy)

  return (d + dt).reshape((-1,1))

def get_hardness_distribution(gtG, max_dist, min_dist, rng, trials, bins, nodes,
                              n_ori, step_size):
  heuristic_fn = lambda node_ids, node_id: \
    heuristic_fn_vec(nodes[node_ids, :], nodes[[node_id], :], n_ori, step_size)
  num_nodes = gtG.num_vertices()
  gt_dists = []; h_dists = [];
  for i in range(trials):
    end_node_id = rng.choice(num_nodes)
    gt_dist = gtG.shortest_distance(source=end_node_id, target=None, 
      max_dist=max_dist, reversed=True)
    gt_dist = np.array(gt_dist.get_array())
    ind = np.where(np.logical_and(gt_dist <= max_dist, gt_dist >= min_dist))[0]
    gt_dist = gt_dist[ind]
    h_dist = heuristic_fn(ind, end_node_id)[:,0]
    gt_dists.append(gt_dist)
    h_dists.append(h_dist)
  gt_dists = np.concatenate(gt_dists)
  h_dists = np.concatenate(h_dists)
  hardness = 1. - h_dists*1./gt_dists
  hist, _ = np.histogram(hardness, bins)
  hist = hist.astype(np.float64)
  hist = hist / np.sum(hist)
  return hist

def rng_next_goal_rejection_sampling(start_node_ids, batch_size, gtG, rng,
                                     max_dist, min_dist, max_dist_to_compute,
                                     sampling_d, target_d,
                                     nodes, n_ori, step_size, bins, M):
  sample_start_nodes = start_node_ids is None
  dists = []; pred_maps = []; end_node_ids = []; start_node_ids_ = [];
  hardnesss = []; gt_dists = [];
  num_nodes = gtG.num_vertices()
  for i in range(batch_size):
    done = False
    while not done:
      if sample_start_nodes:
        start_node_id = rng.choice(num_nodes)
      else:
        start_node_id = start_node_ids[i]
      
      gt_dist = gtG.shortest_distance(source=start_node_id, target=None,
        max_dist=max_dist, reversed=False)
      ind = np.where(np.logical_and(gt_dist <= max_dist, gt_dist >= min_dist))[0]
      ind = rng.permutation(ind)
      gt_dist = gt_dist[ind]*1.
      h_dist = heuristic_fn_vec(nodes[ind, :], nodes[[start_node_id], :],
                                n_ori, step_size)[:,0]
      hardness = 1. - h_dist / gt_dist
      sampled_ind = _rejection_sampling(rng, sampling_d, target_d, bins,
                                        hardness, M)
      if sampled_ind < ind.size:
        # print sampled_ind
        end_node_id = ind[sampled_ind]
        hardness = hardness[sampled_ind]
        gt_dist = gt_dist[sampled_ind]
        done = True

    # Compute distance from end node to all nodes, to return.
    dist, pred_map = gtG.shortest_distance(source=end_node_id, target=None, 
      max_dist=max_dist_to_compute, pred_map=True, reversed=True)
    hardnesss.append(hardness); dists.append(dist); pred_maps.append(pred_map);
    start_node_ids_.append(start_node_id); end_node_ids.append(end_node_id);
    gt_dists.append(gt_dist);
    paths = None
  return start_node_ids_, end_node_ids, dists, pred_maps, paths, hardnesss, gt_dists


def rng_next_goal(start_node_ids, batch_size, gtG, rng, max_dist,
                  max_dist_to_compute, node_room_ids, nodes=None,
                  compute_path=False, dists_from_start_node=None):
  # Compute the distance field from the starting location, and then pick a
  # destination in another room if possible otherwise anywhere outside this
  # room.
  dists = []; pred_maps = []; paths = []; end_node_ids = [];
  for i in range(batch_size):
    room_id = node_room_ids[start_node_ids[i]]
    # Compute distances.
    if dists_from_start_node == None:
      dist = gtG.shortest_distance(source=start_node_ids[i], target=None,
        max_dist=max_dist_to_compute, pred_map=True, reversed=False)
    else:
      dist = dists_from_start_node[i]

    # Randomly sample nodes which are within max_dist.
    near_ids = dist <= max_dist
    near_ids = near_ids[:, np.newaxis]
    # Check to see if there is a non-negative node which is close enough.
    non_same_room_ids = node_room_ids != room_id
    non_hallway_ids = node_room_ids != -1
    good1_ids = np.logical_and(near_ids, np.logical_and(non_same_room_ids, non_hallway_ids))
    good2_ids = np.logical_and(near_ids, non_hallway_ids)
    good3_ids = near_ids
    if np.any(good1_ids):
      end_node_id = rng.choice(np.where(good1_ids)[0])
    elif np.any(good2_ids):
      end_node_id = rng.choice(np.where(good2_ids)[0])
    elif np.any(good3_ids):
      end_node_id = rng.choice(np.where(good3_ids)[0])
    else:
      logging.error('Did not find any good nodes.')

    # Compute distance to this new goal for doing distance queries.
    dist, pred_map = gtG.shortest_distance(
      source=end_node_id, target=None, max_dist=max_dist_to_compute,
      pred_map=True, reversed=True)

    dists.append(dist)
    pred_maps.append(pred_map)
    end_node_ids.append(end_node_id)

    path = None
    if compute_path:
      path = get_path_ids(start_node_ids[i], end_node_ids[i], pred_map)
    paths.append(path)
  
  return start_node_ids, end_node_ids, dists, pred_maps, paths


def rng_room_to_room(batch_size, gtG, rng, max_dist, max_dist_to_compute,
                     node_room_ids, nodes=None, compute_path=False):
  # Sample one of the rooms, compute the distance field. Pick a destination in
  # another room if possible otherwise anywhere outside this room.
  dists = []; pred_maps = []; paths = []; start_node_ids = []; end_node_ids = [];
  room_ids = np.unique(node_room_ids[node_room_ids[:,0] >= 0, 0])
  for i in range(batch_size):
    room_id = rng.choice(room_ids)
    end_node_id = rng.choice(np.where(node_room_ids[:,0] == room_id)[0])
    end_node_ids.append(end_node_id)

    # Compute distances.
    dist, pred_map = gtG.shortest_distance(
      reversed=True, source=end_node_id, target=None, 
      max_dist=max_dist_to_compute, pred_map=True)
    dists.append(dist)
    pred_maps.append(pred_map)

    # Randomly sample nodes which are within max_dist.
    near_ids = dist <= max_dist
    near_ids = near_ids[:, np.newaxis]

    # Check to see if there is a non-negative node which is close enough.
    non_same_room_ids = node_room_ids != room_id
    non_hallway_ids = node_room_ids != -1
    good1_ids = np.logical_and(near_ids, np.logical_and(non_same_room_ids, non_hallway_ids))
    good2_ids = np.logical_and(near_ids, non_hallway_ids)
    good3_ids = near_ids
    if np.any(good1_ids):
      start_node_id = rng.choice(np.where(good1_ids)[0])
    elif np.any(good2_ids):
      start_node_id = rng.choice(np.where(good2_ids)[0])
    elif np.any(good3_ids):
      start_node_id = rng.choice(np.where(good3_ids)[0])
    else:
      logging.error('Did not find any good nodes.')

    start_node_ids.append(start_node_id)

    path = None
    if compute_path:
      path = get_path_ids(start_node_ids[i], end_node_ids[i], pred_map)
    paths.append(path)

  return start_node_ids, end_node_ids, dists, pred_maps, paths


def rng_target_dist_field(batch_size, gtG, rng, max_dist, max_dist_to_compute,
                          nodes=None, compute_path=False):
  # Sample a single node, compute distance to all nodes less than max_dist,
  # sample nodes which are a particular distance away.
  dists = []; pred_maps = []; paths = []; start_node_ids = []
  end_node_ids = rng.choice(gtG.num_vertices(), size=(batch_size,),
                            replace=False).tolist()

  for i in range(batch_size):
    dist, pred_map = gtG.shortest_distance(source=end_node_ids[i], target=None,
      max_dist=max_dist_to_compute, pred_map=True, reversed=True)
    dists.append(dist)
    pred_maps.append(pred_map)

    # Randomly sample nodes which are withing max_dist
    near_ids = np.where(dist <= max_dist)[0]
    start_node_id = rng.choice(near_ids, size=(1,), replace=False)[0]
    start_node_ids.append(start_node_id)

    path = None
    if compute_path:
      path = get_path_ids(start_node_ids[i], end_node_ids[i], pred_map)
    paths.append(path)

  return start_node_ids, end_node_ids, dists, pred_maps, paths
