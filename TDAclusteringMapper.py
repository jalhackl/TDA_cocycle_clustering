from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import tadasets
import networkx as nx
import copy

import pandas as pd
from scipy.ndimage import rotate
import scipy
from scipy import *

from sklearn.base import BaseEstimator, ClusterMixin

#from gtda import *
import umap
from gtda.mapper import *
from gtda.homology import *
from sklearn.cluster import *

import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
import tadasets
import networkx as nx
import copy

import pandas as pd
from scipy.ndimage import rotate
import scipy
from scipy import *


import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

def giotto_color_majority(x):
    """
    Description:
        color nodes according to most frequent label; outputs the index of the most frequent labels, used in TDA mapper for coloring
    Arguments:
        x numpy.ndarray: point cloud data (of a mapper node)
    Returns:
        newvalue numpy.ndarray: index of most frequent label
    """

    counts = np.bincount(x)
    newvalue = np.argmax(counts)
   
    return newvalue


class TDAclusters(BaseEstimator, ClusterMixin):
    """
    Description:
        scikit-learn based clustering class for application of TDA (co)cycle clustering
    """
    
    def __init__(self, 
                 algorithm: str="standard", 
                 ):
        print("init")
        self.algorithm = algorithm
        self.labels_ = None
        
    
    def fit(self, X: np.array, y: np.array=None) -> TDAclusters:
        
       
        labels = np.empty(X.shape[0])
        
        if self.algorithm == "standard":
            #labels = apply_tda_clustering(X, all_get_label=True, include_neighbors=True, barycenter_assignment=False, density_assignment=True, discard_labels=True, discard_clusters=True, min_support_labels=10, min_thresh_labels=0.1, nr_labels=9, max_correl=0.8, min_support_clusters=5, min_thresh_clusters=0.1, nr_clusters=8, nearest_on_trajectory=False, only_decision_labels=True, remove_small=True, birth_weight=1, sub_const=0.001, return_threshes = True, plot_something=False, print_something=False)
            labels = apply_tda_clustering(X)

            labels = np.array(labels)
        elif self.algorithm == "trajectory":
            labels = apply_tda_clustering(X)

            labels = np.array(labels)
            
        self.labels_ = labels

        return self



def run_mapper_with_TDA_clustering(X, filter_f="eccentricity", n_intervals=5, overlap_frac=0.2):
    """
    Description:
        run giotto TDA mapper with scikit-TDA(co)cycle-clustering class
    Arguments:
        x numpy.ndarray: point cloud data
    """
        
    if filter_f == "eccentricity":
        filter_func = Eccentricity()
    else: 
        filter_func = umap.UMAP(n_neighbors=5) 

    # Define cover
    cover = CubicalCover(kind='balanced', n_intervals=n_intervals, overlap_frac=overlap_frac)
    # Choose clustering algorithm 
    clusterer = TDAclusters()
    # Initialise pipeline
    pipe = make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer,
        verbose=True,
        n_jobs=-1,
    )
    # Plot Mapper graph
    fig = plot_static_mapper_graph(pipe, X, color_data=X, node_color_statistic=lambda x: giotto_color_majority(x))
    fig.show(config={'scrollZoom': True})