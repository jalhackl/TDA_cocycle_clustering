
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import tadasets
import plotly.graph_objects as go
import networkx as nx
import copy

from TDAclusteringDatasets import *
from TDAclusteringEvaluation import *
from TDAclusteringMapper import *
from TDAclustering import *
from TDAclusteringUtils import *
from TDAclusteringVisualization import *

def traj_to_graph(xtraj_labeled, zero_breaks=False, zero_value=0):
    """
    Description:
        create a graph in which nodes represent the clustering labels / cognitive states and edges the connections between them
    Arguments:
        xtraj_labeled numpy.ndarray: point cloud data, in the last column the assigned label is stored
    Returns:
        G NetworkX-graph: graph object; nodes correspond to the clustering labels, edges are created when there is a connection along the trajectory
    """
        
    G = nx.DiGraph()
    G.add_node(xtraj_labeled[0][-1])
    last_node = xtraj_labeled[0][-1]
    
    for entry in xtraj_labeled:
        if entry[-1] != last_node and entry[-1] != zero_value:
            G.add_edge(last_node, entry[-1])
            last_node = entry[-1]
    
    return G


def traj_to_graph_check_distance(xtraj_labeled, threshold_distance=None, zero_removed_avg = True, use_zero_reduced_data=True, zero_breaks=False, zero_value=0):
    """
    Description:
        create a graph in which nodes represent the clustering labels / cognitive states and edges the connections between them
    Arguments:
        xtraj_labeled numpy.ndarray: point cloud data, in the last column the assigned label is stored
    Returns:
        G NetworkX-graph: graph object; nodes correspond to the clustering labels, edges are created when there is a connection along the trajectory
    """

    #get average 'velocity' in full trajectory
    x = xtraj_labeled[:,0:-2]
    labels = xtraj_labeled[:,-1]
    x_rem, labels_rem = remove_zero_points(x, labels, zero_value=zero_value)

    if zero_removed_avg == True:
        avg_vel, std_vel = get_avg_velocity_std(x_rem)
    else:
        avg_vel, std_vel = get_avg_velocity_std(x)

    if use_zero_reduced_data == True:
        x = x_rem
        labels = labels_rem
    
        frame = np.array(list(range(len(x))))
        xtraj = np.concatenate((x, np.reshape(frame, (-1, 1))), axis=1)
        xtraj_gt = np.concatenate((xtraj, np.reshape(labels, (-1, 1))), axis=1)
        xtraj_labeled = xtraj_gt
    
    if threshold_distance == None:
        threshold_distance = avg_vel + std_vel
        print("threshold distance:")
        print(threshold_distance)
        
    G = nx.DiGraph()
    G.add_node(xtraj_labeled[0][-1])
    last_node = xtraj_labeled[0][-1]

    last_node_coordinates = xtraj_labeled[0,0:-2]
    
    for entry in xtraj_labeled:
        if entry[-1] != last_node and entry[-1] != zero_value:

            entry_coordinates = entry[0:-2]

            #use euclidean distance
            curr_distance = np.linalg.norm(entry_coordinates - last_node_coordinates)


            if curr_distance < threshold_distance:
                G.add_edge(last_node, entry[-1])

            last_node = entry[-1]

            last_node_coordinates = entry[0:-2]
    
    return G



def traj_to_multigraph(xtraj_labeled, zero_breaks=False, zero_distance = 0, zero_value=0):
    """
    Description:
        create a MultiDiGraph in which nodes represent the clustering labels / cognitive states and the number of edges indicats the number of connections along the trajectory
    Arguments:
        xtraj_labeled numpy.ndarray: point cloud data, in the last column the assigned label is stored
    Returns:
        G NetworkX-MultiDiGraph: graph object; nodes correspond to the clustering labels, edges are created when there is a connection along the trajectory, the number of edges corresponds to the number of connections
    """
        
    G = nx.MultiDiGraph()
    
    for ie, entry in enumerate(xtraj_labeled):
        if xtraj_labeled[ie][-1] != zero_value:
            G.add_node(xtraj_labeled[ie][-1])
            last_node = xtraj_labeled[ie][-1]
            first_nonzero_entry = ie
            break
    
    if zero_breaks == False:
        for entry in xtraj_labeled[first_nonzero_entry:-1]:
            if entry[-1] != last_node and entry[-1] != zero_value:
                G.add_edge(last_node, entry[-1])
                last_node = entry[-1]
    else:
        curr_zero = 0
        for entry in xtraj_labeled[first_nonzero_entry:-1]:
            
            if entry[-1] == zero_value:
                curr_zero = curr_zero + 1
            else:
                curr_zero = 0
            
            if curr_zero >= zero_distance:
            
                if entry[-1] != last_node and entry[-1] != zero_value and last_node !=zero_value:
                    G.add_edge(last_node, entry[-1])
                last_node = entry[-1]
                
            else:
                if entry[-1] != last_node and entry[-1] != zero_value and last_node !=zero_value:
                    G.add_edge(last_node, entry[-1])
                if entry[-1] != last_node and entry[-1] != zero_value:
                    last_node = entry[-1]         
    
    return G


def traj_to_graph_reduced(xtraj_labeled, threshold=3, zero_breaks=False, zero_distance = 0, zero_value=0):
    """
    Description:
        create a directed Graph in which nodes represent the clustering labels / cognitive states and edges the connections between them. At first, a MultiDiGraph is created, afterwards only connections with enough support (i.e. a sufficiently high number of edges) are retained.
    Arguments:
        xtraj_labeled numpy.ndarray: point cloud data, in the last column the assigned label is stored
    Returns:
        G NetworkX-DiGraph: graph object; nodes correspond to the clustering labels, edges are created when there is a connection along the trajectory, the number of edges corresponds to the number of connections
    """
        
    G = nx.MultiDiGraph()
    
    for ie, entry in enumerate(xtraj_labeled):
        if xtraj_labeled[ie][-1] != zero_value:
            G.add_node(xtraj_labeled[ie][-1])
            last_node = xtraj_labeled[ie][-1]
            first_nonzero_entry = ie
            break
    
    
    
    if zero_breaks == False:
        for entry in xtraj_labeled[first_nonzero_entry:-1]:
            if entry[-1] != last_node and entry[-1] != zero_value:
                G.add_edge(last_node, entry[-1])
                last_node = entry[-1]
    else:
        curr_zero = 0
        for entry in xtraj_labeled[first_nonzero_entry:-1]:
            
            if entry[-1] == zero_value:
                curr_zero = curr_zero + 1
            else:
                curr_zero = 0
            
            if curr_zero >= zero_distance:
            
                if entry[-1] != last_node and entry[-1] != zero_value and last_node !=zero_value:
                    G.add_edge(last_node, entry[-1])
                last_node = entry[-1]
                
            else:
                if entry[-1] != last_node and entry[-1] != zero_value and last_node !=zero_value:
                    G.add_edge(last_node, entry[-1])
                if entry[-1] != last_node and entry[-1] != zero_value:
                    last_node = entry[-1]
    
    
    
    edges_list = list(G.edges())
    
    from collections import Counter
    count = Counter(edges_list)
    
    reducedG = nx.DiGraph(G)
    
    for edge in count:
        if count[edge] <= threshold:
            reducedG.remove_edge(edge[0], edge[1])
    
    return reducedG



def paths_calc(G):
    """
    Description:
        the nodes of the input graph are merged: Labels/nodes which have in- and out-degree of one are incoporated into the other states ( which represent 'decision states')
    Arguments:
        G NetworkX-DiGraph: input graph in which nodes represent a single state
    Returns:
        new_graph NetworkX-DiGraph: graph object with merged nodes, so one node represents multiple adjacent states
    """
        
    import copy
    contracted_G = copy.deepcopy(G)
    
    decision_nodes = []
    for node in contracted_G.nodes():
        outdeg = contracted_G.out_degree(node)
        if outdeg > 1:
            decision_nodes.append(node)
    
    all_simple_paths_list = []
    
    all_simple_paths_list_unpruned = []
    for node1 in decision_nodes:
        for node2 in decision_nodes:

            sp = list(nx.all_simple_paths(contracted_G, node1, node2))

            #remove simple paths with additional decision nodes
            valid_simple_path = True

            sp_pruned = []
            for path in sp:
                valid_simple_path = True

                for simple_node in path[1:-1]:

                    if simple_node in decision_nodes:
                        valid_simple_path = False
                        break
                if valid_simple_path == True:
                    sp_pruned.append(path)

            all_simple_paths_list.append(sp_pruned)
            all_simple_paths_list_unpruned.append(sp)

    
    #cycles of length 2
    cycles_length_2 = []
    for node1 in contracted_G.nodes():
        for node2 in contracted_G.nodes():
            if (node2 in contracted_G.neighbors(node1)) and (node1 in contracted_G.neighbors(node2) ):
                cycles_length_2.append([node1, node2])
                        
    new_graph = nx.DiGraph()

    for cycle_nodes in cycles_length_2:
        new_label = ""
        for entry in cycle_nodes:
            new_label = new_label + str(int(entry)) + "-"
        new_label = new_label[:-1]
        new_graph.add_node(new_label, startn = cycle_nodes[0], endn=cycle_nodes[-1])
        
    
    for decision_entry in all_simple_paths_list:
        for entry in decision_entry:
            new_label = ""
            for nname in entry:
                new_label = new_label + str(int(nname)) + "-"
            new_label = new_label[:-1]
            
            new_graph.add_node(new_label, startn=entry[0], endn=entry[-1])
    
    for (node1, attribute1) in new_graph.nodes(data=True):
        for (node2, attribute2) in new_graph.nodes(data=True):
            
            if attribute1['endn'] == attribute2['startn']:
                new_graph.add_edge(node1, node2)
                
    return new_graph



def create_state_diagrams(x, labels, threshold=3, check_distances=False, zero_breaks=False, zero_distance=0, plot_something=True, zero_value=0):
    """
    Description:
        this function takes point cloud data and corresponding labels as input and creates 'cognitive state diagrams'
    Arguments:
        x numpy.ndarray: point cloud data
        labels list: list of corresponding clustering labels
    Returns:
        newpathgraph NetworkX-DiGraph: graph object representing cognitive state diagrams in which one node represents one single state
        new_gt_graph NetworkX-DiGraph: graph object representing cognitive state diagrams with merged nodes, so one node represents multiple adjacent states
    """
       
    frame = np.array(list(range(len(x))))
    xtraj = np.concatenate((x, np.reshape(frame, (-1, 1))), axis=1)
    xtraj_gt = np.concatenate((xtraj, np.reshape(labels, (-1, 1))), axis=1)
    
    if check_distances == False:
        new_gt_graph = traj_to_graph_reduced(xtraj_gt, threshold=threshold, zero_breaks=zero_breaks, zero_distance=zero_distance, zero_value=zero_value)
    else:
        new_gt_graph = traj_to_graph_check_distance(xtraj_gt,  zero_breaks=zero_breaks, zero_value=zero_value)
    #new_gt_multigraph = traj_to_multigraph(xtraj_gt, zero_breaks=zero_breaks, zero_distance=zero_distance, zero_value=zero_value)
    
    if plot_something == True:
        nx.draw(new_gt_graph, with_labels=True)
        plt.show()
        
    newpathgraph = paths_calc(new_gt_graph)
    
    if plot_something == True:
        
        pos=nx.circular_layout(newpathgraph)
        nx.draw(newpathgraph, pos=pos, with_labels=True)
        plt.show()

    return newpathgraph, new_gt_graph



def create_state_diagrams_check_distance(x, labels, threshold=3, threshold_distance=None, zero_breaks=False, zero_distance=0, plot_something=True, zero_value=0):
    """
    Description:
        this function takes point cloud data and corresponding labels as input and creates 'cognitive state diagrams'
    Arguments:
        x numpy.ndarray: point cloud data
        labels list: list of corresponding clustering labels
    Returns:
        newpathgraph NetworkX-DiGraph: graph object representing cognitive state diagrams in which one node represents one single state
        new_gt_graph NetworkX-DiGraph: graph object representing cognitive state diagrams with merged nodes, so one node represents multiple adjacent states
    """
       
    frame = np.array(list(range(len(x))))
    xtraj = np.concatenate((x, np.reshape(frame, (-1, 1))), axis=1)
    xtraj_gt = np.concatenate((xtraj, np.reshape(labels, (-1, 1))), axis=1)

    new_gt_graph = traj_to_graph_check_distance(xtraj_gt, zero_breaks=zero_breaks, threshold_distance=threshold_distance, zero_value=zero_value)
        
    if plot_something == True:
        nx.draw(new_gt_graph, with_labels=True)
        plt.show()
        
    newpathgraph = paths_calc(new_gt_graph)
    
    if plot_something == True:
        
        pos=nx.circular_layout(newpathgraph)
        nx.draw(newpathgraph, pos=pos, with_labels=True)
        plt.show()

    return newpathgraph, new_gt_graph







def compute_gedit_distance(g1, g2, upper_bound=50, timeout=120):
    """
    Description:
        calculation of the graph edit distance for two graphs (e.g. those representing cognitive state diagrams) using a NetworX-function
    Arguments:
        g1 NetworkX-Graph: graph 1 for comparison
        g2 NetworkX-Graph: graph 2 for comparison
    Returns:
        gdist: graph edit distance for the two graph objects
    """

    gdist = nx.graph_edit_distance(g1, g2, upper_bound=upper_bound, timeout=timeout)
    return gdist

