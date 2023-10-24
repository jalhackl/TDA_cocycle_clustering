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
from TDAclustering import *


def show_diagram(diagrams):
    """
    Description:
        plotting of persistence diagrams (created using scikitTDA/ripser)
    Arguments:
        diagrams numpy.ndarray : output of persistent homology function from scikitTDA/ripser
    Returns:
        idx int: index of most persistent homology group in diagrams
    """

    dgm1 = diagrams[1]
    idx = np.argmax(dgm1[:, 1] - dgm1[:, 0])
    plot_diagrams(diagrams, show = False)
    plt.scatter(dgm1[idx, 0], dgm1[idx, 1], 20, 'k', 'x')
    plt.title("Max 1D birth = %.3g, death = %.3g"%(dgm1[idx, 0], dgm1[idx, 1]))
    plt.show()
    
    return idx


def plot_simple(coordinates, labels = None, title = "clustering", remove_0=False, zero_value=0):
    """
    Description:
        static plotting of (2D/3D) point cloud data and clustering labels using matplotlib
    Arguments:
        coordinates numpy.ndarray : point cloud data
        labels list : corresponding clustering labels
    """
        
    if labels==None:
        labels = len(coordinates) * [0]
        
    if remove_0 == True:
        dels = []
        for i, el in enumerate(labels):
            if el == zero_value:
                dels.append(i)

        coordinates = np.delete(coordinates, dels, axis=0) 
        labels=[i for i in labels if i!=zero_value]

    import matplotlib.cm as cm
    coordinates = np.array(coordinates)

    if coordinates.shape[1] == 3:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()
    if coordinates.shape[1] == 3:
        ax.scatter(np.array(coordinates)[:,0], np.array(coordinates)[:,1], np.array(coordinates)[:,2], c=list(labels), cmap=cm.rainbow, s=18)
    else:
        ax.scatter(np.array(coordinates)[:,0], np.array(coordinates)[:,1], c=list(labels), s=18)
    plt.title(title)


def plot_interactive(coordinates, labels, title="clustering", remove_0=False, zero_value=0):
    """
    Description:
        interactive plotting of (3D) point cloud data and clustering labels using plotly
    Arguments:
        coordinates numpy.ndarray : point cloud data
        labels list : corresponding clustering labels
    """
        
    if coordinates.shape[1] !=3:
        print("only for 3D data sets")
        return   
    
    if remove_0 == True:
        dels = []
        for i, el in enumerate(labels):
            if el == zero_value:
                dels.append(i)
        coordinates = np.delete(coordinates, dels, axis=0) 
        labels=[i for i in labels if i!=zero_value]
        
    data=[go.Scatter3d(x=coordinates[:, 0], 
                   y=coordinates[:, 1], 
                   z=coordinates[:, 2],
    mode='lines+markers',
    marker=dict(
        size=5,
        color=labels,                
        colorscale='Viridis',  
        opacity=0.8 ))]
    
    layout = go.Layout(title=title)
    fig = go.Figure(data=data)
    fig.layout = layout
    fig.show()


def plot_pyvista(x, labels):
    """
    Description:
        interactive plotting of (3D) point cloud data and clustering labels using pyvista
    Arguments:
        coordinates numpy.ndarray : point cloud data
        labels list : corresponding clustering labels
    """
    import pyvista as pv
    if labels is not None:
        pv.plot(x, scalars=list(labels), render_points_as_spheres=True)
    else:
        pv.plot(x, render_points_as_spheres=True)


def rainbow_color_gradient(n=10, end=1 / 3):
    """
    Description:
        creation of a color gradient
    Arguments:
        n int : number of colors to generate
        labels list : corresponding clustering labels
    Returns:
        numpy.ndarray: array containing color gradient information
    """
        
    from colorsys import hls_to_rgb

    if n > 1:
        return [hls_to_rgb(end * i / (n - 1), 0.5, 1) for i in range(n)]
    else:
        return [hls_to_rgb(end * i / (n), 0.5, 1) for i in range(n)]
    

def plot_pers_trajectories(subgs, plot_something=True):
    """
    Description:
        plotting and extracting of the induced-subgraph-nodes
    Arguments:
        subgs list : list of NetworkX-graph objects (all the induced subgraphs)
    Returns:
        subarrays list: arrays of nodes in each induced subgraph
    """
        
    subgid = range(len(subgs))
    subarrays = []
    coloredarray = []
    
    if plot_something == True:
        plt.figure()
        plt.title("trajectories/cycles extracted from topological information")
        colors = rainbow_color_gradient(n=len(subgs))
        
    for i, subg in enumerate(subgs):
        subarray = []
        
        if plot_something == True:
            all_pos_path1 = {name:(coordinates[0], coordinates[1]) for (name, coordinates) in nx.get_node_attributes(subg, 'coord').items()}
            nx.draw(subg, all_pos_path1, edge_color=colors[i],  node_size=0) 
            
        for node in subg.nodes(data=False):
            if subg.degree(node) > 0:

                subarray.append(subg.nodes[node]["coord"])
                coloredarray.append([node, subg.nodes[node]["coord"], i])
        subarrays.append(subarray)
    
    if plot_something == True:
        plt.show()
        
    coord = []
    labs = []
    for entry in coloredarray:
        coord.append(entry[1])
        labs.append(entry[2])
    
    if plot_something == True:
        plot_simple(np.array(coord),list(labs), title="trajectories/cycles extracted from topological information")

        if len(coord[0]) == 3:    
            plot_interactive(np.array(coord),list(labs), title="trajectories/cycles extracted from topological information")
    return subarrays






