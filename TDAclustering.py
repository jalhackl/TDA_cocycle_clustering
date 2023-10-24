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
from TDAclusteringStateDiagrams import *
from TDAclusteringUtils import *
from TDAclusteringVisualization import *

def persistent_cocycles_one_hot_paths(D, X, cocycles, diagrams, plot_something=False, include_neighbors=True, thresh_function=None, pers_array="comp_thres", pers_nr=0.2, return_threshes=False):
    """
    Description:
        create one hot encoded labels according to reconstructed cycles.
    Arguments:
        D numpy.ndarray: Genotype matrix from the target population.
        X numpy.ndarray: data points
        cocycles: list of cocycles (computed via scikit-TDA)
        diagrams: information of the persistent homology diagrams (computed via scikit-TDA)
    Returns:
        all_labels numpy.ndarray: one hot encoded label for each datapoint
    """

    dgm1 = diagrams[1]
    
    dia_distances = [i[1] - i[0] for i in dgm1]
    dia_distances = np.array(dia_distances)

    if pers_array == "thres":
        persistence_array =  check_persistence_thresh(dia_distances, pers_nr)
    elif pers_array == "comp_thres":
        persistence_array =  check_persistence_computed_thresh(dia_distances, diff_threshold_fct=thresh_function)
    else:
        persistence_array = check_persistence_nr(dia_distances, clusters = 9)
    
    persistent_cocycles = len(np.where(persistence_array)[0])
    
    max_len_dia = np.argsort([i for i in dia_distances])
    max_len_dia = np.flip(max_len_dia)
    max_len_dia = max_len_dia[0:persistent_cocycles]
    
    print("number of sufficiently persistent homology groups:")
    print(len(max_len_dia))
    
    
    all_labels = np.zeros((X.shape[0], len(max_len_dia)))
    
    if return_threshes == True:
        threshes = []
        death_threshes = []
    
    counter = 0
    for idx in (max_len_dia):
        cocycle = cocycles[1][idx]
        thresh = dgm1[idx, 0]+0.00001 
        
        death_thresh = dgm1[idx, 1]
        
        if return_threshes == True:
            threshes.append(thresh)
            death_threshes.append(death_thresh)

        all_labels = cocycle_clustering_graph_path(D, X, cocycle, thresh, all_labels, None,counter, one_hot=True, include_neighbors=include_neighbors,plot_something=plot_something)
        
        all_labels = all_labels[0]
        counter = counter + 1

    if return_threshes == False:
        return all_labels
    else:
        return all_labels, threshes, death_threshes
    



def cocycle_clustering_graph_path_trajectory_based(D, X, cocycle, thresh, idx, labels=None, id_list=None, new_label=1, one_hot = False, include_neighbors=False, plot_something=True):
    """
    Description:
        reconstruction of cycles; version for trajectory-based algorithm.
    Arguments:
        D numpy.ndarray: Genotype matrix from the target population.
        X numpy.ndarray: data points
        cocycle: cocycle for which the cycle is reconstructed
        thresh: threshold for the current cocycle
        idx: current id
    Returns:
        all_labels numpy.ndarray: one hot encoded label for each datapoint
    """

    import itertools
    
    G = nx.Graph()
    
    N = X.shape[0]

    if id_list == None:
        for i in range(N):
            G.add_node(i, coord=X[i])
    else:
        for i in range(N):
            G.add_node(id_list[i], coord=X[i])
        
    cocycle_nodes = set()
    
    
    #full distance matrix
    from scipy.spatial import distance_matrix
    distance_mat = distance_matrix(X, X)
    
    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                
                vector = G.nodes[j]['coord'] - G.nodes[i]['coord']
                #distance = distance_mat[i][j]
                distance = D[i, j]
                
                G.add_edge(i,j, label="normal_edge", vector=vector, distance = distance)
             
                
    G.remove_edges_from(nx.selfloop_edges(G))
    #Plot cocycle projected to edges under the chosen threshold
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]
        if D[i, j] <= thresh:
            vector = G.nodes[j]['coord'] - G.nodes[i]['coord']
            #distance = distance_mat[i][j]
            distance = D[i, j]
            G.add_edge(i,j, label="cocycle_edge", vector=vector, distance = distance)
            cocycle_nodes.add(i)
            cocycle_nodes.add(j)
            
    
    #remove cocycle_edges again

    clusterset = []
    for cocycle_node in cocycle_nodes:
        cl = nx.node_connected_component(G, cocycle_node)
        clusterset.append(cl)
        
    
    clusters = [item for sublist in clusterset for item in sublist]
    
    
    #create induced subgraph
    persistent_structure = (nx.induced_subgraph(G, clusters)).copy()
    
    
    if plot_something == True:
        pos = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(G, 'coord').items()}

        colors = [G[u][v]['label'] for u,v in G.edges()]
        edge_colors = ['blue' if e=="cocycle_edge" else 'red' for e in colors]

        #nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_size=0)
        #plt.show()
        pos_sub = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(persistent_structure, 'coord').items()}

        colors_sub = [G[u][v]['label'] for u,v in persistent_structure.edges()]
        edge_colors_sub = ['blue' if e=="cocycle_edge" else 'red' for e in colors_sub]

        #nx.draw(persistent_structure, pos_sub, with_labels=True, edge_color=edge_colors_sub, node_size=0)    
        
        
        #nx.draw(persistent_structure, pos_sub,  edge_color=edge_colors_sub, node_size=1)    
        plt.show()
    
     #remove cocycle_edges again
    
    G.remove_edges_from([
    (a, b)
    for a, b, attributes in G.edges(data=True)
    if attributes["label"] == "cocycle_edge"
    ])
    
    cocycle_nodes_list = list(cocycle_nodes)

    cocycle_shortest_path = nx.shortest_path(G, source=cocycle_nodes_list[0], target=cocycle_nodes_list[1])

    pathgraph1 = nx.create_empty_copy(G, with_data=True)
    
    new_p = nx.path_graph(cocycle_shortest_path)
    new_p_edges = new_p.edges()

    new_p_nodes = new_p.nodes()
    
    pathgraph1.add_edges_from(new_p_edges)
    
    #pathgraph clusterlabeling
    nx.set_edge_attributes(pathgraph1, 1, idx)
    
    for node in new_p_nodes:
        pathgraph1.nodes[node][idx] = 1
    
    if plot_something == True:
        pos_path1 = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(pathgraph1, 'coord').items()}
        colors_path1 = [G[u][v]['label'] for u,v in pathgraph1.edges()]

        edge_colors_path1 = ['green' if e=="cocycle_edge" else 'yellow' for e in colors_path1]

    
    #nx.draw(pathgraph1, pos_path1, with_labels=True, edge_color=edge_colors_path1, node_size=0)    
    #plt.show()
    
    all_shortest_paths = nx.all_shortest_paths(G, source=cocycle_nodes_list[0], target=cocycle_nodes_list[1])
    all_pathgraph1 = nx.create_empty_copy(G, with_data=True)
    all_shortest_paths = itertools.islice(all_shortest_paths, 100)
    
    all_path_nodes = set()
    all_path_nodes = {int(a) for b in all_shortest_paths for a in b}
    
    cycle_nodes = (nx.induced_subgraph(G, all_path_nodes)).copy()
    
    
    sumlength = []
    curr_sumlength = 0
    for u,v,a in cycle_nodes.edges(data=True):
        curr_sumlength = curr_sumlength + G[u][v]["distance"]
        sumlength.append(G[u][v]["distance"])
    
   
    curr_sumlength = curr_sumlength/cycle_nodes.number_of_edges()
 
    
    #if plot_something == True:
    for path in all_shortest_paths:

        new_p = nx.path_graph(path)
        new_p_edges = new_p.edges()
        
        new_p_nodes = new_p.nodes()

        all_pathgraph1.add_edges_from(new_p_edges)

        #pathgraph clusterlabeling
        nx.set_edge_attributes(all_pathgraph1, 1, idx)
        
        for node in new_p_nodes:
            all_pathgraph1.nodes[node][idx] = 1
    

    plt.show()
 
    all_neighbor_nodes = []
    
    if include_neighbors == True:
        

        for node in all_path_nodes:
            subgraph = nx.ego_graph(G,node,radius=include_neighbors).copy()
            n = G.neighbors(node)
            nodes_to_be_removed = []
            for neighbor in n:
                if G[node][neighbor]['distance'] > curr_sumlength:
                    nodes_to_be_removed.append(neighbor)
            subgraph.remove_nodes_from(nodes_to_be_removed)

            neighbors= list(subgraph.nodes())

            neighbors = [i for i in neighbors if not i in all_path_nodes]


            all_neighbor_nodes.append(neighbors)

        all_neighbor_nodes = {item for sublist in all_neighbor_nodes for item in sublist}

        
    
    
    if one_hot == False:
        if labels == None:
            labels = [0] * N

    
    all_path_nodes = list(all_path_nodes) #+ list(all_neighbor_nodes) + list(additional_path_nodes)
    
    for i in all_path_nodes:
        if one_hot == False:
            labels[i] = new_label
        else:
            labels[i, new_label] = 1
        
    
    return labels, G, pathgraph1 #all_pathgraph1




def persistent_cocycles_one_hot_paths_trajectory_based(D, X, cocycles, diagrams, thresh_function=None, use_std_for_estimation=False, plot_something=False, set_all=True, include_neighbors=True, label_for_all=True, pers_array="comp_thres", pers_nr=0.2, comb_nr=None, return_threshes=False):
    """
    Description:
        create one hot encoded labels according to reconstructed cycles; version for trajectory-based algorithm..
    Arguments:
        D numpy.ndarray: Genotype matrix from the target population.
        X numpy.ndarray: data points
        cocycles: list of cocycles (computed via scikit-TDA)
        diagrams: information of the persistent homology diagrams (computed via scikit-TDA)
    Returns:
        all_labels numpy.ndarray: one hot encoded label for each datapoint
    """

    
    dgm1 = diagrams[1]
    
    dia_distances = [i[1] - i[0] for i in dgm1]
    dia_distances = np.array(dia_distances)

    if pers_array == "thres":
        persistence_array =  check_persistence_thresh(dia_distances, pers_nr)
    elif pers_array == "comp_thres":
        persistence_array =  check_persistence_computed_thresh(dia_distances, diff_threshold_fct=thresh_function)
    else:
        persistence_array = check_persistence_nr(dia_distances, clusters = 9)
    
    #aber dadurch indices modifiziert - true false array!
    persistent_cocycles = len(np.where(persistence_array)[0])
    
    max_len_dia = np.argsort([i for i in dia_distances])
    max_len_dia = np.flip(max_len_dia)
    max_len_dia = max_len_dia[0:persistent_cocycles]
    
    print("number of sufficiently persistent homology groups:")
    print(len(max_len_dia))
    
    
    all_labels = np.zeros((X.shape[0], len(max_len_dia)))
    
    if return_threshes == True:
        threshes = []
        death_threshes = []
    
    counter = 0
    
    subgraphs = []
    for idx in (max_len_dia):
        cocycle = cocycles[1][idx]
        thresh = dgm1[idx, 0]+0.00001 
        
        death_thresh = dgm1[idx, 1]
        
        if return_threshes == True:
            threshes.append(thresh)
            death_threshes.append(death_thresh)

        all_labels, G, subgraph = cocycle_clustering_graph_path_trajectory_based(D, X, cocycle, thresh, idx, all_labels, None,counter, one_hot=True, include_neighbors=include_neighbors,plot_something=plot_something)
        subgraphs.append(subgraph)
        #all_labels = all_labels[0]
        counter = counter + 1

    #composition of subgraphs
    Gstart = subgraphs[0]
    for subgraph in subgraphs[1:]:
        Gstart = nx.compose(Gstart, subgraph)
        
    if label_for_all == True:
        all_labels = np.zeros((X.shape[0], len(max_len_dia)))
        subarrays = plot_pers_trajectories(subgraphs)
        #hier muesste man wohl mit labels weitermachen
        
            
        from scipy.spatial import distance_matrix
        comb_condition_satisfied = False
        while comb_condition_satisfied == False:    
            if comb_nr == None:
                comb_condition_satisfied = True
            
            extra_vals = []
            for isub, sub in enumerate(subarrays):
                distance_matsub = distance_matrix(np.array(sub), np.array(sub))
                extra_val=np.mean(np.min(np.ma.masked_equal(distance_matsub, 0.0), axis=1))
                extra_vals.append(extra_val)
                
            for ii, i in enumerate(X):
                minvalues = []
                for isub, sub in enumerate(subarrays):
                    
                    distance_mati = distance_matrix([i], np.array(sub))
                    minvalues.append(np.min(distance_mati))

                absolute_min = np.min(minvalues)
                absolute_min_index = np.argmin(minvalues)
                std_min = np.std(minvalues)
                #ja, das legt die 'breite' fest
                #eps = absolute_min + std_min/2
                
                if use_std_for_estimation == True:
                    eps = absolute_min + std_min/2


                for imin, minvalue in enumerate(minvalues):
                    if use_std_for_estimation == False:
                        eps = absolute_min + extra_vals[imin]
                    
                    eps = absolute_min + std_min/2
                    if minvalue <= eps:
                        all_labels[ii, imin] = 1

            
            if comb_nr != None:
                u, c, combs = get_nr_combinations(all_labels)

                if combs > comb_nr:
                    all_labels = np.delete(all_labels, -1, 1)
                    subarrays = subarrays[:-1]
                    if return_threshes == True:
                        threshes = threshes[:-1]
                        death_threshes = death_threshes[:-1]
    
                else:

                    comb_condition_satisfied = True
        
    
    if return_threshes == False:
        return Gstart, subgraphs, all_labels
    else:
        return Gstart, subgraphs, all_labels, threshes, death_threshes




def cocycle_clustering_graph_path(D, X, cocycle, thresh, labels=None, id_list=None, new_label=1, one_hot = False, include_neighbors=False, plot_something=True):
    """
    Description:
        reconstruction of cycles
    Arguments:
        D numpy.ndarray: Genotype matrix from the target population.
        X numpy.ndarray: data points
        cocycle: cocycle for which the cycle is reconstructed
        thresh: threshold for the current cocycle
    Returns:
        labels numpy.ndarray: label according to current cycle
        G NetworX-graph: graph object of all points
        persistent_structure NetworX-graph: graph object of points in current cycle
    """

    import itertools
    
    G = nx.Graph()
    N = X.shape[0]

    if id_list == None:
        for i in range(N):
            G.add_node(i, coord=X[i])
    else:
        for i in range(N):
            G.add_node(id_list[i], coord=X[i])
        
    cocycle_nodes = set()
    

    for i in range(N):
        for j in range(N):
            if D[i, j] <= thresh:
                
                #vector = G.nodes[j]['coord'] - G.nodes[i]['coord']
                distance = D[i, j]
                
                G.add_edge(i,j, label="normal_edge", distance = distance)
             
                
    G.remove_edges_from(nx.selfloop_edges(G))
    #Plot cocycle projected to edges under the chosen threshold
    for k in range(cocycle.shape[0]):
        [i, j, val] = cocycle[k, :]
        if D[i, j] <= thresh:
            #vector = G.nodes[j]['coord'] - G.nodes[i]['coord']
            distance = D[i, j]
            G.add_edge(i,j, label="cocycle_edge", distance = distance)
            cocycle_nodes.add(i)
            cocycle_nodes.add(j)
            
    
    #remove cocycle_edges again
    clusterset = []
    for cocycle_node in cocycle_nodes:
        cl = nx.node_connected_component(G, cocycle_node)
        clusterset.append(cl)
        
    
    clusters = [item for sublist in clusterset for item in sublist]
    
    #create induced subgraph
    persistent_structure = (nx.induced_subgraph(G, clusters)).copy()
    
    
    if plot_something == True:
        pos = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(G, 'coord').items()}

        colors = [G[u][v]['label'] for u,v in G.edges()]
        edge_colors = ['blue' if e=="cocycle_edge" else 'red' for e in colors]

        #nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_size=0)
        #plt.show()
        pos_sub = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(persistent_structure, 'coord').items()}

        colors_sub = [G[u][v]['label'] for u,v in persistent_structure.edges()]
        edge_colors_sub = ['blue' if e=="cocycle_edge" else 'red' for e in colors_sub]

        #nx.draw(persistent_structure, pos_sub, with_labels=True, edge_color=edge_colors_sub, node_size=0)    
        
        nx.draw(persistent_structure, pos_sub,  edge_color=edge_colors_sub, node_size=1)    
        plt.show()
    
     #remove cocycle_edges again
    
    G.remove_edges_from([
    (a, b)
    for a, b, attributes in G.edges(data=True)
    if attributes["label"] == "cocycle_edge"
    ])
    
    cocycle_nodes_list = list(cocycle_nodes)

    cocycle_shortest_path = nx.shortest_path(G, source=cocycle_nodes_list[0], target=cocycle_nodes_list[1])

    pathgraph1 = nx.create_empty_copy(G, with_data=True)
    
    new_p = nx.path_graph(cocycle_shortest_path)
    new_p_edges = new_p.edges()

    pathgraph1.add_edges_from(new_p_edges)
    
    if plot_something == True:
        pos_path1 = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(pathgraph1, 'coord').items()}
        colors_path1 = [G[u][v]['label'] for u,v in pathgraph1.edges()]

        edge_colors_path1 = ['green' if e=="cocycle_edge" else 'yellow' for e in colors_path1]

    
    #nx.draw(pathgraph1, pos_path1, with_labels=True, edge_color=edge_colors_path1, node_size=0)    
    #plt.show()
    
    all_shortest_paths = nx.all_shortest_paths(G, source=cocycle_nodes_list[0], target=cocycle_nodes_list[1])
    all_pathgraph1 = nx.create_empty_copy(G, with_data=True)
    all_shortest_paths = itertools.islice(all_shortest_paths, 100)
    
    all_path_nodes = set()
    all_path_nodes = {int(a) for b in all_shortest_paths for a in b}
    
    cycle_nodes = (nx.induced_subgraph(G, all_path_nodes)).copy()
    
    
    sumlength = []
    curr_sumlength = 0
    for u,v,a in cycle_nodes.edges(data=True):
        curr_sumlength = curr_sumlength + G[u][v]["distance"]
        sumlength.append(G[u][v]["distance"])
   
    curr_sumlength = curr_sumlength/cycle_nodes.number_of_edges()
    
    if plot_something == True:
        for path in all_shortest_paths:

            new_p = nx.path_graph(path)
            new_p_edges = new_p.edges()

            all_pathgraph1.add_edges_from(new_p_edges)
    
        
    additional_path_nodes = []

    if plot_something == True:    
        all_pos_path1 = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(all_pathgraph1, 'coord').items()}
        all_colors_path1 = [G[u][v]['label'] for u,v in all_pathgraph1.edges()]
        all_edge_colors_path1 = ['yellow' if e=="cocycle_edge" else 'green' for e in all_colors_path1]
   
        #nx.draw(all_pathgraph1, all_pos_path1, with_labels=True, edge_color=all_edge_colors_path1, node_size=0)  


    plt.show()
 
    all_neighbor_nodes = []
    
    if include_neighbors == True:
        
        for node in all_path_nodes:
            subgraph = nx.ego_graph(G,node,radius=include_neighbors).copy()
            n = G.neighbors(node)
            nodes_to_be_removed = []
            for neighbor in n:
                if G[node][neighbor]['distance'] > curr_sumlength:
                    nodes_to_be_removed.append(neighbor)
            subgraph.remove_nodes_from(nodes_to_be_removed)

            neighbors= list(subgraph.nodes())
            neighbors = [i for i in neighbors if not i in all_path_nodes]
            all_neighbor_nodes.append(neighbors)

        all_neighbor_nodes = {item for sublist in all_neighbor_nodes for item in sublist}
    
    if one_hot == False:
        if labels == None:
            labels = [0] * N

    all_path_nodes = list(all_path_nodes) + list(all_neighbor_nodes) + list(additional_path_nodes)
    
    for i in all_path_nodes:
        if one_hot == False:
            labels[i] = new_label
        else:
            labels[i, new_label] = 1
    
    return labels, G, persistent_structure



def apply_tda_clustering(x, all_get_label=False, num_voxels_per_axis = 10, min_points_per_voxel=5, point_fraction=0.1, target_r=False, target_radius_fraction=0.01, thresh_function=None, include_neighbors=True, low_density_removal=False, smoothing=False, smoothing_but_return_full=True, downsampling=False, downsampling_but_return_full=True, barycenter_assignment=True, density_assignment=False, discard_labels=True, nr_labels=None, discard_clusters=True, min_support_labels=10, min_thresh_labels=0.1, max_correl=0.85, min_support_clusters=10, min_thresh_clusters=0.1, nr_clusters=None, nearest_on_trajectory=False, only_decision_labels=True, remove_small=False, birth_weight=1, sub_const=0, return_threshes = True, plot_something=True, print_something=True, zero_value=0):
    """
    Description:
        main function, application of TDA (co)cycle clustering
    Arguments:
        x numpy.ndarray: data points
    Returns:
        labels numpy.ndarray: label for each data point
    """

    if smoothing == True:
        if x.shape[1] == 3:
            original_x = x.copy()
            x = downsampling_smoothing_points(x, num_voxels_per_axis = num_voxels_per_axis, min_points_per_voxel=min_points_per_voxel)
            #if print_something == True:
            print("number of interpolated points:")
            print(len(x))
        else:
            print("function only available for 3D data")
    
    if low_density_removal == True:
        new_x, rem_indices = density_selection(x)
        old_x=x.copy()
        x=new_x.copy()
        
    if downsampling == True:

        downsampled_x, discarded_indices = downsampling_points(x, point_fraction=point_fraction, target_r=target_r, target_radius_fraction=target_radius_fraction)

        not_downsampled_x = x.copy()
        x = downsampled_x.copy()

        print("number of downsampled points:")
        print(len(x))
        
    
    D, cocycles, diagrams = compute_persistent_homology(x)
    
    if return_threshes == False:
        labels = persistent_cocycles_one_hot_paths(D, x, cocycles, diagrams, plot_something=plot_something, thresh_function=thresh_function, include_neighbors=include_neighbors, return_threshes = return_threshes)
    else:
        labels, threshes, death_threshes = persistent_cocycles_one_hot_paths(D, x, cocycles, diagrams, plot_something=plot_something, thresh_function=thresh_function,include_neighbors=include_neighbors, return_threshes = return_threshes)

    retained_indices = np.array(list(range(labels.shape[1])))
    if discard_labels == True:
        labels, retained_indices = remove_unimportant_labels(labels, min_support=min_support_labels, min_thresh=min_thresh_labels, max_correl=max_correl, nr_labels=nr_labels)
    
    if only_decision_labels == True:
        old_labels = labels.copy()
        #not used anymore
        #labels = extract_decision_labels(labels)
        
        labels = label_zero_regions(x, labels, old_labels, threshes, death_threshes, retained_indices, birth_weight=birth_weight, sub_const=sub_const)
    
    labels = one_hot_to_int(labels)
    labels = get_low_number_list(labels)
    
    if discard_clusters == True:
        labels = remove_unimportant_clusters(labels, min_support=min_support_clusters, min_thresh=min_thresh_clusters, nr_clusters=nr_clusters)
        labels = get_low_number_list(labels)
        
        
    if downsampling == True:
        if downsampling_but_return_full == True:
            indsort = np.sort(discarded_indices)
            for index in indsort:
                x = np.insert(x, index, not_downsampled_x[index], axis=0)  
                labels.insert(index, zero_value)
        else:
            return [x,labels]
        
    if low_density_removal == True:
        iindpx = np.where((rem_indices==True))
        indsort = np.sort(iindpx)[0]
        for index in indsort:
            x = np.insert(x, index, old_x[index], axis=0)  
            labels.insert(index, zero_value)
            #janochmalaebdern
        
    if all_get_label == True:
        labels = get_final_label_for_remaining(x,labels,barycenter_assignment=barycenter_assignment,density_assignment=density_assignment,nearest_on_trajectory=nearest_on_trajectory)
    
    if remove_small == True:
        labels = replace_small_states(x, labels, min_size=None)
        
    if smoothing == True and x.shape[1] == 3:
        if smoothing_but_return_full == True:
            labels = assign_not_interpolated_points(original_x, x, labels, min_pts=1, zero_value=0)
        else:
            return [x, labels]
        
    return labels



def apply_tda_clustering_trajectories(x, all_get_label=False, thresh_function=None, num_voxels_per_axis = 10, min_points_per_voxel=5, point_fraction=0.1, target_r=False, target_radius_fraction=0.01, include_neighbors=True, low_density_removal=False, smoothing=False, smoothing_but_return_full=True, downsampling=False, downsampling_but_return_full=True, discard_labels=True, comb_nr=None, discard_clusters=False, min_support_labels=10, min_thresh_labels=0.25, max_correl=0.85, min_support_clusters=10, min_thresh_clusters=0.1, nr_clusters=None, nearest_on_trajectory=False, only_decision_labels=True, remove_small=False, birth_weight=1, sub_const=0, return_threshes = True, plot_something=True, print_something=True, zero_value=0):
    """
    Description:
        main function, application of TDA (co)cycle clustering; trajectory-based version
    Arguments:
        x numpy.ndarray: data points
    Returns:
        biggraph NetworkX-graph: graph containing all data points and reconstructed cycles
        subgs NetworkX-graph: graph containing one single cycle and its points
        labels numpy.ndarray: label for each data point
    """

    if smoothing == True:
        if x.shape[1] == 3:
            original_x = x.copy()
            x = downsampling_smoothing_points(x, num_voxels_per_axis=num_voxels_per_axis, min_points_per_voxel=min_points_per_voxel)
            #if print_something == True:
            print("number of interpolated points:")
            print(len(x))
        else:
            print("function only available for 3D data")
    
    if low_density_removal == True:
        new_x, rem_indices = density_selection(x)
        old_x=x.copy()
        x=new_x.copy()
        
    if downsampling == True:
        downsampled_x, discarded_indices = downsampling_points(x, point_fraction=point_fraction, target_r=target_r, target_radius_fraction=target_radius_fraction)
        not_downsampled_x = x.copy()
        x = downsampled_x.copy()
        #if print_something == True:
        print("number of downsampled points:")
        print(len(x))
    
    D, cocycles, diagrams = compute_persistent_homology(x)
    
    
    biggraph, subgs, labels, threshes, death_threshes = persistent_cocycles_one_hot_paths_trajectory_based(D, x, cocycles, diagrams, thresh_function=thresh_function, comb_nr=comb_nr, plot_something=plot_something, include_neighbors=include_neighbors, return_threshes = return_threshes)
    
    
    retained_indices = np.array(list(range(labels.shape[1])))
    if discard_labels == True:
        labels, retained_indices = remove_unimportant_labels(labels, min_support=min_support_labels, min_thresh=min_thresh_labels, max_correl=max_correl)
    
    if only_decision_labels == True:
        old_labels = labels.copy()
        #not used anymore
        #labels = extract_decision_labels(labels)
        
        labels = label_zero_regions(x, labels, old_labels, threshes, death_threshes, retained_indices, birth_weight=birth_weight, sub_const=sub_const)
    
    labels = one_hot_to_int(labels)
    labels = get_low_number_list(labels)
    
    if discard_clusters == True:
        labels = remove_unimportant_clusters(labels, min_support=min_support_clusters, min_thresh=min_thresh_clusters, nr_clusters=nr_clusters)
        labels = get_low_number_list(labels)
        
    if downsampling == True:
        if downsampling_but_return_full == True:
            indsort = np.sort(discarded_indices)
            for index in indsort:
                x = np.insert(x, index, not_downsampled_x[index], axis=0)  
                labels.insert(index, zero_value)
        else:
            return [x,labels]
        
    if low_density_removal == True:
        iindpx = np.where((rem_indices==True))
        indsort = np.sort(iindpx)[0]
        for index in indsort:
            x = np.insert(x, index, old_x[index], axis=0)
            labels.insert(index, zero_value)
        
    if all_get_label == True:
        labels = get_final_label_for_remaining(x,labels,nearest_on_trajectory=nearest_on_trajectory)
        
    if remove_small == True:
        labels = replace_small_states(x, labels, min_size=None)
        
    if smoothing == True and x.shape[1] == 3:
        if smoothing_but_return_full == True:
            labels = assign_not_interpolated_points(original_x, x, labels, min_pts=1, zero_value=zero_value)
        else:
            return [biggraph, subgs, labels, x]
    
    return biggraph, subgs, labels



def apply_tda_clustering_iter(x, all_get_label=True, include_neighbors=True, barycenter_assignment=True, density_assignment=False, discard_labels=True, discard_clusters=True, min_support_labels=10, min_thresh_labels=0.1, max_correl=0.85, min_support_clusters=10, min_thresh_clusters=0.1, nr_clusters=8, nearest_on_trajectory=False, only_decision_labels=True, remove_small=False, return_threshes = True, plot_something=True, print_something=True):
    """
    Description:
        main function, application of TDA (co)cycle clustering; iterated version
    Arguments:
        x numpy.ndarray: data points
    Returns:
        x_labels_list numpy.ndarray: final label for each data point
    """

    x_new = x.copy()
    x_thinned = x.copy()
    x_labels_list = []

    size_does_not_change_anymore = False
    nr_iter = 0
    while size_does_not_change_anymore == False:
        if nr_iter > 0:
            zero_indices = [i for i, v in enumerate(list(labels_new)) if v!=0]
            labels_thinned = [v for v in list(labels_new) if v==0]
            x_thinned = np.delete(x_new, zero_indices, 0)

        D, cocycles, diagrams = compute_persistent_homology(x_thinned)


        nr_iter = nr_iter + 1
        if nr_iter > 1 and (len(x_new) == len(x_thinned) or len(diagrams[1]) == 0):
            size_does_not_change_anymore = True

        else:
            x_new = x_thinned.copy()

            if return_threshes == False:
                labels_new = persistent_cocycles_one_hot_paths(D, x_thinned, cocycles, diagrams, plot_something=plot_something, include_neighbors=include_neighbors, return_threshes = return_threshes)
            else:
                labels_new, threshes, death_threshes = persistent_cocycles_one_hot_paths(D, x_thinned, cocycles, diagrams, plot_something=plot_something, include_neighbors=include_neighbors, return_threshes = return_threshes)
            
            if only_decision_labels == True:
                old_labels = labels_new.copy()
                retained_indices = np.array(list(range(labels_new.shape[1])))
                labels_new = label_zero_regions(x_thinned, labels_new, old_labels, threshes, death_threshes, retained_indices, birth_weight=1, sub_const=0)


            labels_new = one_hot_to_int(labels_new)
            labels_new = get_low_number_list(labels_new)

            plot_simple(x_thinned, list(labels_new), remove_0=True)
            plt.show()
            x_labels_list.append([x_thinned.copy(), labels_new.copy()])


    return x_labels_list



def apply_tda_clustering_iter_trajectories(x, return_full=False, all_get_label=True, include_neighbors=True, barycenter_assignment=True, density_assignment=False, discard_labels=True, discard_clusters=True, min_support_labels=10, min_thresh_labels=0.1, max_correl=0.85, min_support_clusters=10, min_thresh_clusters=0.1, nr_clusters=8, nearest_on_trajectory=False, only_decision_labels=True, remove_small=False, return_threshes = True, plot_something=True, print_something=True):
    """
    Description:
        main function, application of TDA (co)cycle clustering; iterated and tajectory-based version
    Arguments:
        x numpy.ndarray: data points
    Returns:
        x_labels_list numpy.ndarray: final label for each data point
    """
    
    x_new = x.copy()
    x_thinned = x.copy()
    x_labels_list = []
    
    size_does_not_change_anymore = False
    nr_iter = 0
    while size_does_not_change_anymore == False:
        if nr_iter > 0:
            zero_indices = [i for i, v in enumerate(list(labels_new)) if v!=0]
            labels_thinned = [v for v in list(labels_new) if v==0]
            x_thinned = np.delete(x_new, zero_indices, 0)

        if len(x_thinned) < 3:
            size_does_not_change_anymore = True
            break

        D, cocycles, diagrams = compute_persistent_homology(x_thinned)

        nr_iter = nr_iter + 1
        if nr_iter > 1 and (len(x_new) == len(x_thinned) or len(diagrams[1]) == 0):
            size_does_not_change_anymore = True
            break
        else:
            x_new = x_thinned.copy()

            biggraph, subgs, labels_new, threshes, death_threshes = persistent_cocycles_one_hot_paths_trajectory_based(D, x_thinned, cocycles, diagrams, comb_nr=None,label_for_all=False, plot_something=plot_something, include_neighbors=include_neighbors, return_threshes = return_threshes)
            
            if only_decision_labels == True:
                old_labels = labels_new.copy()
                retained_indices = np.array(list(range(labels_new.shape[1])))
                labels_new = label_zero_regions(x_thinned, labels_new, old_labels, threshes, death_threshes, retained_indices, birth_weight=1, sub_const=0)


            labels_new = one_hot_to_int(labels_new)
            labels_new = get_low_number_list(labels_new)

            plt.show()
            plot_simple(x_thinned, list(labels_new), remove_0=True)
            plt.show()
            x_labels_list.append([x_thinned.copy(), labels_new.copy()])
            
    return x_labels_list



