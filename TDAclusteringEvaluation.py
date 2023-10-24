from sklearn.cluster import *
import pandas as pd
import sklearn

import TDAclustering

from TDAclusteringDatasets import *
from TDAclustering import *
from TDAclusteringMapper import *
from TDAclusteringStateDiagrams import *
from TDAclusteringUtils import *
from TDAclusteringVisualization import *

def compare_clusterings(x, true_labels, methods=["tda","kmeans", "dbscan"], metrics=["rand", "adjusted_rand"], additional_labels=None, additional_methods=None, nr_clusters=8, plot_something=True, return_dict=False):
    """
    Description:
        comparison of different clustering methods; "tda" corresponds to TDA (co)cycle clustering
    Arguments:
        x numpy.ndarray: point cloud data
        true_labels list: true labels of x
    Returns:
        results_list list: list with metrics for each method 
        labels_dict dict: labels, keys correspond to methods
    """
    
    import itertools
    
    labels_dict = dict()
    
    results_list = []
    
    labels_dict["true"] = true_labels

    if additional_labels != None:
        if additional_methods == None: 
            additional_methods = ["method " + str(x) for x in range(len(additional_labels))]

        for ia, additional_label in enumerate(additional_labels):
            labels_dict[additional_methods[ia]] = additional_label
    
    for method in methods:
        
        if method == "tda":
            new_labels = apply_tda_clustering(x, nr_clusters=nr_clusters)
            labels_dict["tda"] = new_labels
            
        if method == "kmeans" and nr_clusters != None:
            new_labels = KMeans(n_clusters=8,  n_init="auto").fit(x).labels_
            labels_dict["kmeans"] = new_labels
            
        if method == "dbscan":
            new_labels = DBSCAN(eps=0.3, min_samples=2).fit(x).labels_
            labels_dict["dbscan"] = new_labels
            
            
    res = list(itertools.combinations(labels_dict, 2))
    
    for pair in res:
            
        for metric in metrics:
            if metric == "rand":
                met_result = sklearn.metrics.rand_score(labels_dict[pair[0]], labels_dict[pair[1]])                
                results_list.append([pair[0], pair[1], metric, met_result])
                
            if metric == "adjusted_rand":
                met_result = sklearn.metrics.adjusted_rand_score(labels_dict[pair[0]], labels_dict[pair[1]])                
                results_list.append([pair[0], pair[1], metric, met_result])

    if return_dict == True:
        return results_list, labels_dict        
    else:    
        return results_list
    

def compare_state_diagrams(x, true_labels, predicted_labels_array, methods_array=None, true_zero=-1, predicted_zero=0, threshold=1, plot_something=True):
    """
    Description:
        comparison of state diagrams induced by different clustering methods
    Arguments:
        x numpy.ndarray: point cloud data
        true_labels list: true labels of x
        predicted_labels_array list: labels predicted by different methods
    Returns:
        gedit_dict dict: graph edit distances, keys correspond to methods
    """
        
    true_newpathgraph, true_new_gt_graph = create_state_diagrams(x, true_labels, threshold=threshold, plot_something=plot_something, zero_value=true_zero)
    
    if methods_array == None:
        methods_array = ["prediction " + str(x) for x in range(len(predicted_labels_array))]
    
    gedit_dict = dict()
    
    for ip, prediction in enumerate(predicted_labels_array):
            predicted_newpathgraph, predicted_new_gt_graph = create_state_diagrams(x, prediction, threshold=threshold, plot_something=plot_something, zero_value=predicted_zero)
            
            new_gedit = compute_gedit_distance(true_newpathgraph, predicted_newpathgraph)
            
            gedit_dict[methods_array[ip]] = new_gedit
            
    return gedit_dict





def compare_clusterings_gedit(x, true_labels, additional_labels=None, additional_methods=None, methods=["tda","kmeans", "dbscan"], metrics=["rand", "adjusted_rand"], nr_clusters=8, threshold=3,  diagrams=True, plot_something=True):
    """
    Description:
        comparison of different clustering methods; clustering results as well as induced state diagrams are compared; "tda" corresponds to TDA (co)cycle clustering
    Arguments:
        x numpy.ndarray: point cloud data
        true_labels list: true labels of x
    Returns:
        results_list list: clustering results for each method
        gedit_list list: graph edit distances for each method
    """
    
    import itertools
    
    labels_dict = dict()
    results_dict = dict()
    
    results_list = []
    
    labels_dict["true"] = true_labels
    
    if additional_labels != None:
        if additional_methods == None:
            method_nr = list(range(len(additional_labels)))
            additional_methods = ["method" + str(j) for j in method_nr]
        
        for i, entry in enumerate(additional_labels):
            labels_dict[additional_methods[i]] = entry
    
            
    
    for method in methods:
        
        if method == "tda":
            #new_labels = apply_tda_clustering(x, all_get_label=True, barycenter_assignment=False, min_support_labels=30, density_assignment=True,nr_clusters=8,remove_small=True, plot_something=False)
            new_labels = apply_tda_clustering(x, all_get_label=True, barycenter_assignment=False, min_support_labels=10, density_assignment=True,low_density_removal=False, smoothing=False,nr_clusters=8,remove_small=True, birth_weight=1,nearest_on_trajectory=False,plot_something=False)
            labels_dict["tda"] = new_labels
            
        if method == "kmeans" and nr_clusters != None:
            new_labels = KMeans(n_clusters=8,  n_init="auto").fit(x).labels_
            labels_dict["kmeans"] = new_labels
            
        if method == "dbscan":
            new_labels = DBSCAN(eps=0.3, min_samples=2).fit(x).labels_
            labels_dict["dbscan"] = new_labels
            
    if plot_something == True:
        for entry in labels_dict:
            plot_simple(x,list(labels_dict[entry]), title=entry)
            plt.show()
            
            
    if diagrams == True:
        gedit_list = []
        states_dict = dict()
        for i, entry in enumerate(labels_dict):
            
            if entry == "true":
                new_diagram = create_state_diagrams(x, labels_dict[entry], zero_breaks=False, threshold=threshold,zero_value=-1)
                true_diagram = new_diagram
            else:
                new_diagram = create_state_diagrams(x, labels_dict[entry], zero_breaks=False, threshold=threshold,zero_value=0)
                
            states_dict[entry] = new_diagram[1]
        for entry in states_dict:
            if entry != "true":

                gedit = compute_gedit_distance(states_dict[entry], true_diagram[1])
                gedit_list.append([entry, gedit])

            
    res = list(itertools.combinations(labels_dict, 2))
    
    for pair in res:
            
        for metric in metrics:
            if metric == "rand":
                met_result = sklearn.metrics.rand_score(labels_dict[pair[0]], labels_dict[pair[1]])                
                results_list.append([pair[0], pair[1], metric, met_result])
                
            if metric == "adjusted_rand":
                met_result = sklearn.metrics.adjusted_rand_score(labels_dict[pair[0]], labels_dict[pair[1]])                
                results_list.append([pair[0], pair[1], metric, met_result])
         
    if diagrams == False:
        return results_list
    else:
        return results_list, gedit_list
    

def compare_state_diagrams(x, true_labels, predicted_labels_array, methods_array=None, true_zero=-1, predicted_zero=0, threshold=1, plot_something=True):
    true_newpathgraph, true_new_gt_graph = create_state_diagrams(x, true_labels, threshold=threshold, plot_something=plot_something, zero_value=true_zero)
    
    if methods_array == None:
        methods_array = ["prediction " + str(x) for x in range(len(predicted_labels_array))]
    
    gedit_dict = dict()
    
    for ip, prediction in enumerate(predicted_labels_array):
            predicted_newpathgraph, predicted_new_gt_graph = create_state_diagrams(x, prediction, threshold=threshold, plot_something=plot_something, zero_value=true_zero)
            
            new_gedit = compute_gedit_distance(true_newpathgraph, predicted_newpathgraph)
            
            gedit_dict[methods_array[ip]] = new_gedit
            
    return gedit_dict



def compare_state_diagrams_check_distance(x, true_labels, predicted_labels_array, methods_array=None, true_zero=-1, predicted_zero=0, threshold=1, plot_something=True):
    """
    Description:
        comparison of state diagrams induced by different clustering methods
    Arguments:
        x numpy.ndarray: point cloud data
        true_labels list: true labels of x
        predicted_labels_array list: labels predicted by different methods
    Returns:
        gedit_dict dict: graph edit distances, keys correspond to methods
    """
        
    true_newpathgraph, true_new_gt_graph = create_state_diagrams(x, true_labels, threshold=threshold, plot_something=plot_something, zero_value=true_zero)
    
    if methods_array == None:
        methods_array = ["prediction " + str(x) for x in range(len(predicted_labels_array))]
    
    gedit_dict = dict()
    
    for ip, prediction in enumerate(predicted_labels_array):
            predicted_newpathgraph, predicted_new_gt_graph = create_state_diagrams_check_distance(x, prediction, threshold=threshold, plot_something=plot_something, zero_value=predicted_zero)
            
            new_gedit = compute_gedit_distance(true_newpathgraph, predicted_newpathgraph)
            
            gedit_dict[methods_array[ip]] = new_gedit
            
    return gedit_dict
    


def various_tda_computations(x):
    """
    Description:
        compute TDA (co)cycle clustering with different parameters for a point cloud data set
    Arguments:
        x numpy.ndarray: point cloud data
    Returns:
        collected_labels list: list containing the predicted labels by each applied method
        method_strings list: list containing the parameters of the applied methods
    """
        
    method_strings = []
    collected_labels = []
    
    new_labels = apply_tda_clustering(x, all_get_label=True, barycenter_assignment=False, min_support_labels=10, density_assignment=True,low_density_removal=False, smoothing=True,nr_clusters=8,remove_small=True, nearest_on_trajectory=False,plot_something=False)
    st = "tda, density, 8 clusters, rem_small"
    
    collected_labels.append(new_labels)
    method_strings.append(st)
    
    new_labels2 = apply_tda_clustering(x, all_get_label=True, barycenter_assignment=True, min_support_labels=10, density_assignment=False,low_density_removal=False, smoothing=True,nr_clusters=8,remove_small=True, nearest_on_trajectory=False,plot_something=False)
    st2 = "tda, barycener, 8 clusters, rem_small"
    
    collected_labels.append(new_labels2)
    method_strings.append(st2)
    
    return collected_labels, method_strings

