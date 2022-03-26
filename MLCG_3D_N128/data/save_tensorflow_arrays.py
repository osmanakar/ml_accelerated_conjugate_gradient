import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf 
import gc
import scipy.sparse as sparse
#import matplotlib.pyplot as plt

project_name = "MLCG_3D_N128"
project_folder_subname = os.path.basename(os.getcwd())
print("project_folder_subname = ", project_folder_subname)
project_folder_general = "/home/oak/projects/ML_preconditioner_project/"+project_name+"/"
project_data_folder = "/home/oak/projects/ML_preconditioner_project/data/"+project_name+"/"


sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf

dim = 128
dim2 = dim**3

#%% Creating ConjugateGradientSparse Object
print("Creating ConjugateGradientSparse Object")

#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
#pres_lap = pl.pressure_laplacian(dim-1)
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)
CG = cg.ConjugateGradientSparse(A_sparse)

#%% Data Loading 2
#d_name = "b_rhs_10000_1_10000_faulty_ritz_vectors_V2_for_3D_random_N"
d_name = "b_rhs_10000_1_10000_faulty_ritz_vectors_V2_for_3D_reshaped_random_N"
print("Loading ... "+d_name)
#b_rhs_10000_1_10000_faulty_ritz_vectors_V2_for_3D_random_N127
#d_name = "b_rhs_10000_eigvector_equidistributed_random_N"
with open(project_data_folder+d_name+str(dim-1)+'.npy', 'rb') as f:  
    b_rhs = np.load(f)

print(np.linalg.norm(b_rhs)**2)
with open(project_data_folder+'test_idx_10000'+'.npy', 'rb') as f:
    test_idx = np.load(f)
with open(project_data_folder+'train_idx_10000'+'.npy', 'rb') as f:
    train_idx = np.load(f)
print(b_rhs.shape)

test_idx = test_idx[0:500]
train_idx = train_idx[0:4500]

x_train = tf.convert_to_tensor(b_rhs[train_idx],dtype=tf.float32) #140GB
x_test = tf.convert_to_tensor(b_rhs[test_idx],dtype=tf.float32) #16GB        
del b_rhs
gc.collect(generation=2)

#%%
import pickle
filename = project_data_folder + d_name + str(dim-1) + "_"
with open(filename, "wb") as f:
    pickle.dump([x_train,x_test], f)
    
#%%
"""
with open(filename, "rb") as f:
    a1,a2 = pickle.load(f)
"""















