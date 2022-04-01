import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf 
import gc
import time
import scipy.sparse as sparse

project_name = "tests"
project_folder_general = os.path.dirname(os.path.realpath(__file__))+"/"

#%%
sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import matplotlib.pyplot as plt
import helper_functions as hf

#%% Creating ConjugateGradientSparse Object
print("Creating ConjugateGradientSparse Object")
dim = 128
dim2 = dim**3
#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
#pres_lap = pl.pressure_laplacian(dim-1)
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)
CG = cg.ConjugateGradientSparse(A_sparse)

#%%












