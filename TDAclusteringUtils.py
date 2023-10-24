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
from TDAclustering import *
from TDAclusteringVisualization import *

def compute_persistent_homology(x):
    """
    Description:
        computation of persistent homology (using scikit-TDA/ripser)
    Arguments:
        x numpy.ndarray: data points
    Returns:
        D
        cocycles
        diagrams
    """
        
    result = ripser(x, do_cocycles=True)
    diagrams = result['dgms']
    cocycles = result['cocycles']
    D = result['dperm2all']
    
    return D, cocycles, diagrams


def check_persistence_nr(distances, clusters = 5):
    """
    Description:
        choose the most persistent homology groups depending on the specified number of clusters
    Arguments:
        distances numpy.ndarray: numpy array with lifetimes of the determined homology groups
    Returns:
        persistent numpy.ndarray: boolean array specifying which homology groups are sufficiently persistent (True) or not (False)
    """
    
    max_len_dia = np.argsort([i for i in distances])
    
    upper_bound = distances[max_len_dia[-clusters]]

    persistent = (distances >= upper_bound)

    return persistent




def check_persistence_thresh(distances, threshold=0.1):
    """
    Description:
        choose the most persistent homology groups depending on the specified threshold
    Arguments:
        distances numpy.ndarray: numpy array with lifetimes of the determined homology groups
    Returns:
        persistent numpy.ndarray: boolean array specifying which homology groups are sufficiently persistent (True) or not (False)
    """

    maxdist = np.max(distances)
       
    persistent = (distances >= threshold*maxdist)

    return persistent



def check_persistence_computed_thresh(distances, diff_threshold_fct=None, plot_dists=False):
    """
    Description:
        choose the most persistent homology groups depending on the specified method; by default, otsu's method is applied
    Arguments:
        distances numpy.ndarray: numpy array with lifetimes of the determined homology groups
    Returns:
        persistent numpy.ndarray: boolean array specifying which homology groups are sufficiently persistent (True) or not (False)
    """
    
    if np.ptp(distances) < np.mean(distances):
        print("no calculation of otsu threshold")
        return np.full_like(distances, True)
    
    if diff_threshold_fct == None:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(distances)
    elif diff_threshold_fct == "multiotsu":
        from skimage.filters import threshold_multiotsu
        thresh = threshold_multiotsu(distances)[-1]
    else:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(distances)
    
    if plot_dists == True:
        print(thresh)
        plt.hist(distances)
        plt.show()
        
    persistent = (distances >= thresh)

    return persistent



def create_point_cloud_from_nx(G):
    """
    Description:
        choose the most persistent homology groups depending on the specified method; by default, otsu's method is applied
    Arguments:
        G NetworkX-graph: input graph
    Returns:
        indices list: list containing the indices of the data points (nodes of the input graph)
        coordinates list: list containing the coordinates of the data points (nodes of the input graph)
    """
    
    a = nx.get_node_attributes(G,'coord')
    indices = []
    coordinates = []
    for entry in a:

        indices.append(entry)
        coordinates.append(a[entry])

    return indices, coordinates



def one_hot_to_int(one_hot):
    """
    Description:
        transform the one hot encoded labeling into a cluster (integer) value
    Arguments:
        one_hot numpy.ndarray: ont hot encoded labels for all data points
    Returns:
        labels_int list: list containing the clusters
    """
    
    labels_int = []
    for label in one_hot:
        value_int = int(''.join([str(int(x)) for x in label]))
        
        labels_int.append(value_int)
    return labels_int



def get_low_number_list(labels, zero_value=0, print_something=False):
    """
    Description:
        change the cluster labels in such a way that the start at 1 and proceed as 2,3,4...
    Arguments:
        labels list: list containing the clusters
    Returns:
        final_labels list: list containing the clusters as ascending integers starting from 1
    """
        
    unique_list = list(dict.fromkeys(labels))
    unique_list.sort()
    
    if print_something==True:
        print("show unique label list")
        print(unique_list)
    
    if zero_value in unique_list:
        new_label_nr = list(range(len(unique_list)))
    else:
        new_label_nr = list(range(1,len(unique_list)+1))
    
    replacement_dict =  dict(zip(unique_list, new_label_nr))
    
    final_labels = list(map(replacement_dict.get, labels, labels))

    return final_labels




def remove_unimportant_labels(labels, nr_labels = None, min_support=10, min_thresh=0.1,max_correl=0.85, store_indices=True, print_something=True):
    """
    Description:
        remove 'unimportant' one hot encoded labels (e.g. labels with low support)
    Arguments:
        labels numpy.ndarray: one hot encoded labels
    Returns:
        labels numpy.ndarray: one hot encoded labels with 'unimportant' ones set to zero
    """


    if store_indices == True:
        indices_array = np.array(list(range(labels.shape[1])))
        indices_full_array = indices_array.copy()
    
    if print_something == True:
        print("nr. of labels before removal of unimportant labels:")
        print(labels.shape[1])
        
        
    labelsT = labels.T
    columns_to_be_removed=set()
    all_one_counts = []
    for column in labelsT:
        unique_values, counts = np.unique(column, return_counts=True)
        all_one_counts.append(counts[1])
        
    maxone = max(all_one_counts)
    maxindex = all_one_counts.index(max(all_one_counts))
                
    if min_support != None:
        for iv, value in enumerate(all_one_counts):
            if value < min_support:
                columns_to_be_removed.add(iv)            
    
    if min_thresh != None:
        for iv, value in enumerate(all_one_counts):
            if value < min_thresh * maxone:
                columns_to_be_removed.add(iv)            
            
    if len(list(columns_to_be_removed)) > 0:
        labels = np.delete(labels,np.array(list(columns_to_be_removed)), axis=1)
    
    if store_indices == True:
        indices_array = [indices_array[i] for i in range(len(indices_array)) if i not in list(columns_to_be_removed)]
    
    #find highly correlated labels
    import pandas as pd
    
    df = pd.DataFrame(labels)
    corr_matrix = df.corr().abs()
    
    upper_diag = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))

    columns_to_be_removed_corr = [column for column in upper_diag.columns if any(upper_diag[column] > max_correl)]

    df.drop(columns_to_be_removed_corr, axis=1, inplace=True)
    
    if store_indices == True:
        indices_array = [indices_array[i] for i in range(len(indices_array)) if i not in list(columns_to_be_removed_corr)]
    
    labels = np.array(df)
    
    
    if nr_labels != None:
        comb_condition_satisfied = False
        while comb_condition_satisfied == False:

                u, c, combs = get_nr_combinations(labels)

                if combs > nr_labels:
                    labels = np.delete(labels, -1, 1)
                    if store_indices == True:
                        indices_array = indices_array[:-1]

                else:

                    comb_condition_satisfied = True

    
    
    if print_something == True:
        print("nr. of labels after removal of unimportant labels:")
        print(labels.shape[1])
        
    if store_indices == False:
        return labels
    else:
        return labels, indices_array#, retained_indices
    



def remove_unimportant_clusters(int_labels, min_support=10, min_thresh=0.1, nr_clusters=None, print_something=True):
    """
    Description:
        remove 'unimportant' clusters (e.g. clusters with low support)
    Arguments:
        int_labels list: list with cluster values
    Returns:
        int_labels list: list with cluster values, 'unimportant' ones set to zero
    """
    
    unique_values, counts = np.unique(int_labels, return_counts=True)
    
    if print_something == True:
        print("nr. of labels before removal of unimportant clusters:")
        print(len(unique_values))
    
    max_value = max(unique_values)
    max_index = int_labels.index(max(unique_values))
    
    clusters_to_be_removed=set()
    
    for ic, count in enumerate(counts):
        if min_support != None:
            if count < min_support:
                clusters_to_be_removed.add(ic)
        if min_thresh != None:
            if count < min_thresh * max_value:
                clusters_to_be_removed.add(ic)
                
    int_labels = [0 if nr in clusters_to_be_removed else nr for nr in int_labels ]

    unique_values, counts = np.unique(int_labels, return_counts=True)
    if print_something == True:
        print("nr. of labels after removal of unimportant clusters:")
        print(len(unique_values))
        
    if nr_clusters != None:
        max_value = max(unique_values)
        max_index = int_labels.index(max(unique_values))
        
        count_sort_ind = np.argsort(-counts)
        sufficient_clusters = unique_values[count_sort_ind[0:nr_clusters+1]]
        
        int_labels = [0 if nr not in sufficient_clusters else nr for nr in int_labels ]
        
        if print_something == True:
            print("nr. of labels after removal of unimportant clusters by specifying nr. of clusters:")
            print(len(unique_values))
        
    
    return int_labels




def extract_decision_labels(labels, print_something=True):
    """
    Description:
        a simple (especially for 3D cases too simple) function for extracting 'decision labels' (or rather setting non-decision labels to zero): If there are two areas participating in multiple homology groups and an connecting area participating in only one, the latter is set to zero
    Arguments:
        labels numpy.ndarray: one hot encoded labels
    Returns:
        labels numpy.ndarray: one hot encoded labels, labels for data points in non-decision regions set to zero
    """

    for ir, row in enumerate(labels):
        if (len([i for i in row if i!=0])) < 2:
            labels[ir] = [0 for i in row]
            
    return labels



def get_velocities(x):
    """
    Description:
        calculating the distances between adjacent data points along the trajectory
    Arguments:
        x numpy.ndarray: one hot encoded labels
    Returns:
        velocities list: one hot encoded labels, labels for data points in non-decision regions set to zero
    """
        
    velocities = []
    for i, pos in enumerate(x[0:-1]):
        velocity = np.linalg.norm(x[i+1] - x[i])
        velocities.append(velocity)
    return velocities


def get_avg_velocity(x):
    """
    Description:
        calculating the average distance between data points along the trajectory
    Arguments:
        x numpy.ndarray: point cloud data
    Returns:
        velocities float: average distance
    """
        
    velocities = get_velocities(x)
    return np.mean(velocities)

def get_avg_velocity_std(x):
    """
    Description:
        calculating the average distance and std between data points along the trajectory
    Arguments:
        x numpy.ndarray: point cloud data
    Returns:
        float: average distance
        float: standard deviation

    """
        
    velocities = get_velocities(x)
    return np.mean(velocities), np.std(velocities)

def get_nr_combinations(labels):
    """
    Description:
        get the number of labels, i.e. the number of unique one hot encoded combinations
    Arguments:
        labels numpy.ndarray: one hot encoded labels
    Returns:
        u float: unique values
        c int: count vector
        len(c) int: length of c
    """
        
    u, c = np.unique(labels, return_counts=True, axis=0)
    return u, c, len(c)


def grav_center(points):
    """
    Description:
        calculating the barycenter of the points
    Arguments:
        points numpy.ndarray: point cloud data
    Returns:
        gravcenter numpy.ndarray: coordinates of barycenter
    """

    avgv = len(points)

    gravcenter = np.zeros((len(points[0])))
    for i, point in enumerate(points):
        for j in range(len(point)):
            gravcenter[j] = gravcenter[j] + point[j]
    
            
    for i, gravc in enumerate(gravcenter):
        gravcenter[i] = gravc / avgv

    return gravcenter   


def grav_center_per_label(x, all_labels):
    """
    Description:
        calculating the barycenter of the points grouped by each label
    Arguments:
        x numpy.ndarray: point cloud data
        all_labels list: labels assigned to points
    Returns:
        gravs list: coordinates of barycenter for each label
    """
        
    unique_labels, ct = np.unique(all_labels, return_counts=True)
    points_per_label_array = []
    for label in unique_labels:
        indices = [i for i in range(len(all_labels)) if all_labels[i] == label]
        points_per_label = x[indices]
        points_per_label_array.append(points_per_label)
        
    gravs = dict()
    for ip, points in enumerate(points_per_label_array):
        grav = grav_center(points)
        
        gravs[unique_labels[ip]] = grav
    return gravs


def get_state_length(x, all_labels):
    """
    Description:
        returning the length of all state segments along the trajectory
    Arguments:
        x numpy.ndarray: point cloud data
        all_labels list: labels assigned to points
    Returns:
        state_lengths list: coordinates of barycenter for each label
    """
        
    frame = np.array(list(range(len(x))))
    xtraj = np.vstack((frame, all_labels)).T
    curr_state = xtraj[0][-1]
    curr_state_length = 0
    
    state_lengths = []
    
    for ip, point in enumerate(xtraj):
        if xtraj[ip][-1] == curr_state:
            curr_state_length = curr_state_length + 1
        else:
            state_lengths.append(curr_state_length)
            curr_state = xtraj[ip][-1]
            curr_state_length = 1
            
    return state_lengths



def replace_small_states(x, all_labels, min_size=5):
    """
    Description:
        replace small state segments along the trajectory
    Arguments:
        x numpy.ndarray: point cloud data
        all_labels list: labels assigned to points
    Returns:
        all_labels list: labels assigned to points, small state segments are replaced
    """
    
    if min_size == None:
        sl = get_state_length(x, all_labels)
        min_size = int(np.quantile(sl, 0.25))
    
    frame = np.array(list(range(len(x))))
    xtraj = np.vstack((frame, all_labels)).T
    
    curr_state = xtraj[0][-1]
    previous_state = curr_state
    previous_state_length = 0
    curr_state_length = 0
    start_index = 0
    
    state_lengths = []
    
    all_labels = np.array(all_labels)
    for ip, point in enumerate(xtraj):
        if xtraj[ip][-1] == curr_state:
            curr_state_length = curr_state_length + 1
        else:
            if curr_state_length < min_size:
                all_labels[start_index:ip+1] = previous_state
                curr_state = previous_state
                curr_state_length = curr_state_length + previous_state_length
            
            else:
                previous_state = curr_state
                previous_state_length = curr_state_length
                curr_state = xtraj[ip][-1]
                start_index = ip
                curr_state_length = 1
    all_labels = list(all_labels)       
    return all_labels


def get_final_label_for_remaining(X,all_labels, id_list=None, nearest_on_trajectory=False, barycenter_assignment=False, density_assignment=False, curr_cutoff=None, min_pts=5, nearest_on_trajectory_start=False, nearest_start_max=5, zero_value=0):
    """
    Description:
        endow all zero-labeled data point with a cluster label
    Arguments:
        X numpy.ndarray: point cloud data
        all_labels list: labels assigned to points
    Returns:
        all_labels list: labels assigned to points, every point has a non-zero cluster label
    """

    if nearest_on_trajectory_start == True:
        N = X.shape[0]
                
        frame = np.array(list(range(len(x))))
        xtraj = np.vstack((frame, all_labels)).T
        all_labels = np.array(all_labels)
        
        
        for ip, point in enumerate(xtraj[1:]):
            ip2 = ip+1
            if xtraj[ip2][-1] == zero_value and xtraj[ip2-1][-1] != zero_value:
                cluster_found = False
                step = 0
                while cluster_found == False:
                    if ip2+step < len(frame):
                        if xtraj[ip2+step][-1] != zero_value:
                            cluster_found = True
                            next_value = xtraj[ip2+step][-1] 

                    step=step+1
                    if step > nearest_start_max:
                        break
                    if cluster_found == True:
                        patch_size = step
                        all_labels[ip2:ip2+patch_size+1] = next_value
                        xtraj[ip2:ip2+patch_size+1] = next_value
        all_labels = list(all_labels)
                        
    
    
    if barycenter_assignment == True:
        gravs = grav_center_per_label(X, all_labels)
        
        for il, label in enumerate(all_labels):
            if label == zero_value:
                minvalue = float('inf')
                for ik, key in enumerate(gravs):
                    if key != zero_value:
                        barycenter = gravs[key]
                        dist = np.linalg.norm(X[il] - barycenter)
                        if dist < minvalue:
                            minvalue = dist
                            minindex = key
                all_labels[il] = minindex
                
        return all_labels
                    
    if density_assignment == True:

        return ball_tree_assignment(X, all_labels, zero_value=zero_value)
                
        
    #in principle, one could also combine both approaches, e.g. assign points with very near neighbors by density assignment and the rest based on the trajectory
    if nearest_on_trajectory == False:
        G = nx.Graph()
        N = X.shape[0]

        if id_list == None:
            for i in range(N):
                G.add_node(i, coord=X[i], cluster=all_labels[i])
        else:
            for i in range(N):
                G.add_node(id_list[i], id_index=i, coord=X[i], cluster=all_labels[i])

        #full distance matrix
        from scipy.spatial import distance_matrix
        distance_mat = distance_matrix(X, X)
        for i in range(N):
            for j in range(N):

                #vector = G.nodes[j]['coord'] - G.nodes[i]['coord']
                distance = distance_mat[i][j]

                G.add_edge(i,j, label="normal_edge", distance = distance)

        unlabeled_nodes = [x for x,y in G.nodes(data=True) if y['cluster']==zero_value]


        new_label_set = True
        new_labeled_dict = dict()
        newly_labeled_nodes = []
        while len([value for value in unlabeled_nodes if value not in newly_labeled_nodes]) != 0:

            new_label_set = False
            for node in unlabeled_nodes:
                if node not in newly_labeled_nodes:
                    next_cluster_found = False
                    curr_node = node
                    further_nodes = []
                    further_nodes.append(node)
                    processed_nodes = []
                    while next_cluster_found == False: #and curr_node not in processed_nodes:
                        neighbors = G.neighbors(curr_node)
                        mindist = float("inf")
                        for i, neighbor in enumerate(neighbors):
                            if (G[curr_node][neighbor]["distance"] < mindist and neighbor not in processed_nodes and neighbor not in further_nodes): 
                                mindist = G[curr_node][neighbor]["distance"]
                                #minindex = i
                                minneighbor = neighbor

                        if G.nodes[minneighbor]["cluster"] != zero_value:
                            new_label_set = True
                            next_cluster_found = True
                            mincluster = G.nodes[minneighbor]["cluster"] 
                            #further_nodes = []

                            for further_node in further_nodes:
                                if G.nodes[further_node]["cluster"] == zero_value:
                                    nx.set_node_attributes(G, {further_node:mincluster}, 'cluster')
                                    newly_labeled_nodes.append(further_node)
                                    new_labeled_dict[further_node] = mincluster
                            if G.nodes[curr_node]["cluster"] == zero_value:
                                nx.set_node_attributes(G, {curr_node:mincluster}, 'cluster')
                                new_labeled_dict[curr_node] = mincluster
                        else:
                            further_nodes.append(curr_node)
                            processed_nodes.append(curr_node)
                            curr_node = minneighbor


        #extract cluster
        if id_list == None:
            for key, value in new_labeled_dict.items():
                all_labels[key] = value

        return all_labels
    
    else:
        
        N = X.shape[0]
                
        frame = np.array(list(range(len(X))))
        xtraj = np.vstack((frame, all_labels)).T

        for ip, point in enumerate(xtraj):
            if xtraj[ip][-1] == zero_value:
                clusterfound = False
                step = 0
                while clusterfound == False:
                    if ip+step < len(frame):
                        if xtraj[ip+step][-1] != zero_value:
                            clusterfound = True
                            #xtraj[ip][-1] = xtraj[ip+step][-1]
                            all_labels[ip] = xtraj[ip+step][-1]
                            break
                    if ip-step >= 0:
                        if xtraj[ip-step][-1] != zero_value:
                            clusterfound = True
                            #xtraj[ip][-1] = xtraj[ip-step][-1]
                            all_labels[ip] = xtraj[ip-step][-1]
                            break
                    step=step+1
                    
        return all_labels
    



def label_zero_regions_thresh_fct(curr_thresh, curr_death_thresh, birth_weight=1, sub_const=0):
    """
    Description:
        compute the current threshold specifying under which conditions points are assigned to the same non-decision region
    Arguments:
        curr_thresh int: current birthtime
        curr_death_thresh int: current deathtime
    Returns:
        float: current final threshold
    """
        
    death_weight = 1 - birth_weight
    return curr_thresh*birth_weight+curr_death_thresh*death_weight-sub_const


def label_zero_regions(x, new_labels, old_labels, threshes, death_threshes, retained_indices, birth_weight=1, sub_const=0, min_comp_size=6, id_list = None, plot_something=False):
    """
    Description:
        label all non decision regions (i.e. points between decision regions); in fact, look for each unique combination if it has to be split further
    Arguments:
        x numpy.ndarray: point cloud data
        all_labels numpy.ndarray: labels assigned to points
        old_labels numpy.ndarray: labels assigned to points 
        threshes list
        death_threshes list
        retained_indices list
    Returns:
        new_labels list: labels assigned to points, all 'non-decision regions' have a distinct labeling
    """
        
    zero_array = []
    
    #get all u
    all_unique_indices = []
    all_uniques = np.unique(old_labels, axis=0)
    all_uniques = [x for x in all_uniques if 1 in np.unique(x)]

    for iu, unique_label in enumerate(all_uniques):
        unique_indices = [idx for idx, value in enumerate(old_labels) if list(value) == list(unique_label)]
        all_unique_indices.append(unique_indices)
        
    #determine max tresh
    max_thresh_indices = []
    
    max_threshes = []
    for ie, entry in enumerate(all_uniques):
        max_thresh = -np.inf
        max_thresh_ind = -1
        for iel, el in enumerate(entry):
            if entry[iel] == 1:
                #curr_death = death_threshes[retained_indices[iel]] #- threshes[retained_indices[iel]]
                curr_death = threshes[retained_indices[iel]]
                if curr_death > max_thresh:
                    max_thresh = curr_death
                    max_thresh_ind = iel
        max_thresh_indices.append(max_thresh_ind)
        max_threshes.append(max_thresh)
        

    for ie, entry_unique in enumerate(all_uniques):
        
        #curr_thresh = threshes[retained_indices[max_thresh_indices[ie]]]
        curr_death_thresh  = death_threshes[retained_indices[max_thresh_indices[ie]]]
        

        curr_thresh = max_threshes[ie]
        
        entry = all_unique_indices[ie]
            
        G = nx.Graph()

        if id_list == None:
            for i in entry:
                G.add_node(i, coord=x[i])
        else:
            for i in entry:
                G.add_node(id_list[i], coord=x[i])

        #full distance matrix
        from scipy.spatial import distance_matrix
        distance_mat = distance_matrix(x, x)

        
        for i in entry:
            for j in entry:
                #if distance_mat[i, j] <= curr_thresh:
                if distance_mat[i, j] <= label_zero_regions_thresh_fct(curr_thresh, curr_death_thresh, birth_weight=birth_weight, sub_const=sub_const):
                #if distance_mat[i, j] <= (curr_death_thresh):

                    #vector = G.nodes[j]['coord'] - G.nodes[i]['coord']
                    distance = distance_mat[i, j]

                    G.add_edge(i,j, label="normal_edge", distance = distance)


        G.remove_edges_from(nx.selfloop_edges(G))

        compnr = 0
        #if the components are separated by a 'decision component', they will get a different combination of labels

        for component in nx.connected_components(G):

            indu_subg = G.subgraph(component).copy()
            if len(indu_subg) < min_comp_size:
                continue

            new_column = np.zeros(len(old_labels))

            new_column[list(component)] = 1

            new_labels = np.append(new_labels, np.reshape(new_column, (-1, 1)), axis=1)
            compnr = compnr + 1

            if plot_something == True:
                plt.show()
                nx.draw(G.subgraph(component).copy())
                plt.show()
                 
    return new_labels



def irq_outliers(x):
    """
    Description:
        find outliers using irq
    Arguments:
        x numpy.ndarray: data values (in the given context, usually specifying the persistence time of homology groups)
    Returns:
        indices numpy.ndarray: indices of non-outliers
    """
        
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)

    iqr = q3 - q1

    threshold = 0.5 * iqr

    indices = np.where((x < q1 - threshold))
    
    return indices


def density_selection(x, bandwidth = 1, kernel = 'epanechnikov', threshold_method="thres", max_thres=0.1, plot_densities=True):
    """
    Description:
        select data points within regions of sufficient density using a density kernel
    Arguments:
        x numpy.ndarray: point cloud data
    Returns:
        persistent_x numpy.ndarray: points within regions of sufficient density
        rem_indices numpy.ndarray: indices of removed points
    """
        
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(x)
    densities = kde.score_samples(x)

    if plot_densities == True:

        plt.hist(densities)
        plt.show()
        plt.boxplot(densities)
        plt.show()
        
    irq_indices = irq_outliers(densities)
    
    mask = np.ones(densities.size, dtype=bool)
    mask[irq_indices] = False
    persistent_x = x[mask]
    
  
    rem_indices = ~mask
    
    
    return persistent_x, rem_indices



def downsampling_smoothing_points(x, num_voxels_per_axis = 10, min_points_per_voxel=5):
    """
    Description:
        create a downsampled and smoothed data set
    Arguments:
        x numpy.ndarray: point cloud data
    Returns:
        x_interpolated numpy.ndarray: interpolated (smoothed / downasmpled) point cloud data
    """

    import point_cloud_utils as pcu
    
    bbox_size = x.max(0) - x.min(0)

    sizeof_voxel = bbox_size / num_voxels_per_axis

    if min_points_per_voxel == None:
        x_interpolated = pcu.downsample_point_cloud_on_voxel_grid(sizeof_voxel, x)
    else:
        x_interpolated = pcu.downsample_point_cloud_on_voxel_grid(sizeof_voxel, x, min_points_per_voxel=min_points_per_voxel)
    
    return x_interpolated



def downsampling_points(x, point_fraction=0.1, target_r=False, target_radius_fraction=0.01):
    """
    Description:
        create a downsampled data set; in contrast to downsampling_smoothing_points, the remaining data points are points of the original set
    Arguments:
        x numpy.ndarray: point cloud data
    Returns:
        x_downsampled numpy.ndarray: downsampled point cloud data
        not_idx numpy.ndarray: indices of removed data points
    """

    import point_cloud_utils as pcu
    
    if target_r == False:
        target_num_pts= int(point_fraction*x.shape[0]) 
        idx = pcu.downsample_point_cloud_poisson_disk(x, num_samples=target_num_pts)

    else:
        target_radius = np.linalg.norm(x.max(0) - x.min(0)) * target_radius_fraction  
        idx = pcu.downsample_point_cloud_poisson_disk(x, -1, radius=target_radius)

    #sort the ids of the points remaining in the downsampled set
    idx = np.sort(idx)
        
    x_downsampled = x[idx]

    not_idx = [i for i in range(len(x)) if i not in idx]

    return x_downsampled, not_idx


def ball_tree_assignment(x, curr_labels, min_pts=5, zero_value=0):
    """
    Description:
        apply the BallTree algorithm to all data points with zero (i.e. unassigned) clustering label
    Arguments:
        x numpy.ndarray: point cloud data
        curr_labels numpy.ndarray: labels of point cloud data
    Returns:
        new_labels numpy.ndarray: labels of point cloud data, hitherto unassigned points are assigned according to BallTree algorithm
    """

    from sklearn.neighbors import BallTree

    curr_labels_list = list(curr_labels)
    curr_labels_ar = np.array(curr_labels)
    
    zero_indices = [i for i in range(len(curr_labels)) if curr_labels[i] == zero_value]
    
    labels_wo = [i for i in curr_labels_list if i != zero_value]
    labels_wo = np.array(labels_wo)
    x_wo = [i for ii, i in enumerate(x) if curr_labels_list[ii] != zero_value]
    x_only = [i for ii, i in enumerate(x) if curr_labels_list[ii] == zero_value] 
    
    labels_u = np.unique(curr_labels_ar)
    
    tree = BallTree(x_wo)
    
    new_labels = curr_labels_list.copy()
    for i,zero_index in enumerate(zero_indices):
        curr_value = [x[zero_index]]
                         
        dist, ind = tree.query(curr_value, k=min_pts)                
        ind = ind.astype(int)
 
        nearest_points_labels = labels_wo[ind[0]]
        
        unique, counts = np.unique(nearest_points_labels, return_counts=True)
        index = np.argmax(counts)
        new_cluster = unique[index]
        
        new_labels[zero_index] = new_cluster
        
    return new_labels



def assign_not_interpolated_points(original_x, interpolated_x, interpolated_labels, min_pts=1, zero_value=0):
    """
    Description:
        for application of TDA on smoothed/interpolated data: apply the BallTree algorithm to all original data points assigning them to the next cluster of the interpolated data set
    Arguments:
        original_x numpy.ndarray: original point cloud data
        interpolated_x numpy.ndarray: interpolated/smoothed/downsampled point cloud data
        interpolated_labels list: clustering labels of interpolated/smoothed/downsampled data points
    Returns:
        all_labels list: final clustering labels
    """
        
    from sklearn.neighbors import BallTree
    tree = BallTree(interpolated_x)
    
    interpolated_labels=np.array(interpolated_labels)
    
    all_labels = []
    for point in original_x:
        dist, ind = tree.query([point], k=min_pts)  
        ind = ind.astype(int)

        nearest_points_labels = interpolated_labels[ind[0]]
        
        unique, counts = np.unique(nearest_points_labels, return_counts=True)
        index = np.argmax(counts)
        new_cluster = unique[index]
        
        all_labels.append(new_cluster)
        
    return all_labels


def remove_zero_points(x, labels, zero_value=0):
    """
    Description:
        remove all data points without assigned clustering label (i.e. with 'zero_value') from point cloud data set
    Arguments:
        x numpy.ndarray: original point cloud data
        labels numpy.ndarray: clustering labels of original point cloud data
    Returns:
        x_rem numpy.ndarray: final clustering labels
        labels_rem numpy.ndarray: indices of removed data points
    """

    zero_indices = [i for i, value in enumerate(labels) if value == zero_value]
    
    x_rem = np.delete(x, zero_indices, axis=0) 
    labels_rem = [labels[i] for i in range(len(labels)) if i not in zero_indices]
    
    return x_rem, labels_rem 