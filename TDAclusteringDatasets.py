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

from TDAclustering import *
from TDAclusteringEvaluation import *
from TDAclusteringMapper import *
from TDAclusteringStateDiagrams import *
from TDAclusteringUtils import *
from TDAclusteringVisualization import *


def load_worm_data(points_file=None, labels_file=None, worm_nr="worm_0"):
    
    if points_file != None and labels_file != None:
        df = pd.read_csv(points_file, header=None, sep=" ", names=["x","y","z"])
        labels = pd.read_csv(labels_file, header=None, dtype= int, sep=" ", names=["label"])
    else:
        df = pd.read_csv('Y0_tr__AbCNet_' + worm_nr + '.csv', header=None, sep=" ", names=["x","y","z"])
        labels = pd.read_csv('B_train_1__AbCNet_' + worm_nr + '.csv', header=None, dtype= int, sep=" ", names=["label"])

    x = df.to_numpy()
    labels_list = labels.to_numpy()
    labels_list = labels_list.flatten()

    return x, labels_list


def create_3_cycles(n=315, d=1, r=4, noise=0.0001):
    x = tadasets.dsphere(n=n, d=d, r=r,noise=noise)
    xb = tadasets.dsphere(n=n, d=d, r=r, noise=noise)
    xc = tadasets.dsphere(n=n, d=d, r=r, noise=noise)

    xb[:,0] = xb[:,0] + 2*r
    xc[:,0] = xc[:,0] + 4*r

    x  = np.concatenate((x,xb))
    x  = np.concatenate((x,xc))

    return x


def create_5_cycles(n=315, d=1, r=4, noise=0.0001):

    x = tadasets.dsphere(n=n, d=d, r=r,noise=noise)
    xb = tadasets.dsphere(n=n, d=d, r=r, noise=noise)
    xc = tadasets.dsphere(n=n, d=d, r=r, noise=noise)

    xd = tadasets.dsphere(n=n, d=d, r=r, noise=noise)
    xe = tadasets.dsphere(n=n, d=d, r=r, noise=noise)

    xb[:,0] = xb[:,0] + 2*r
    xc[:,0] = xc[:,0] + 4*r

    xd[:,0] = xd[:,0] + 2*r
    xe[:,0] = xe[:,0] + 2*r
    xd[:,1] = xd[:,1] + 2*r
    xe[:,1] = xe[:,1] - 2*r

    x  = np.concatenate((x,xb))
    x  = np.concatenate((x,xc))
    x  = np.concatenate((x,xd))
    x  = np.concatenate((x,xe))

    return x


def create_2_overlapping_cycles():
    
    x = tadasets.dsphere(n=315, d=1, r=4,noise=0.0001)
    xc = tadasets.dsphere(n=315, d=1, r=7, noise=0.0001)
    xc[:,0] = xc[:,0] + 8
    x  = np.concatenate((x,xc))

    plt.scatter(x[:, 0], x[:, 1])
    plt.axis('equal')
    plt.show()

def rotateMatrixX(x):
    return np.array([[1,0,0],[0,np.cos(x), -np.sin(x)], [0,np.sin(x), np.cos(x)]])

def rotateMatrixZ(x):
    return np.array([[np.cos(x), -np.sin(x),0], [np.sin(x), np.cos(x),0], [0,0,1]])

def rotateMatrix(x):
    return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])


def create_2_infinity_signs(n=515, noise=0.0001, rotation_rad=1.5708):
    x = tadasets.infty_sign(n=n,  noise=noise)

    add = np.ones((len(x),1))
    x = np.hstack((x,add))
    xc = tadasets.infty_sign(n=n, noise=noise)
    xc[:,0] = xc[:,0] #+ 8

    add = np.ones((len(xc),1))
    xc = np.hstack((xc,add))
    xc = xc @ rotateMatrixX(rotation_rad).T
    x  = np.concatenate((x,xc))

    return x



def create_2_overlapping_infinity_signs(n=515, noise=0.0001, rotation_rad=1.5708):

    x = tadasets.infty_sign(n=n,  noise=noise)
    xc = tadasets.infty_sign(n=n, noise=noise)
    xc[:,0] = xc[:,0] #+ 8

    xc = xc @ rotateMatrix(rotation_rad).T

    x  = np.concatenate((x,xc))
    add = np.array(len(x) * [1])
    add = np.ones((len(x),1))

    x = np.hstack((x,add))


    plt.scatter(x[:, 0], x[:, 1], x[:, 2])
    plt.axis('equal')
    plt.show()

    return x



