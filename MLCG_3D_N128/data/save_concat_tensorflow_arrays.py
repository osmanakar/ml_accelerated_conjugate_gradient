import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf 
import gc
import scipy.sparse as sparse
import pickle

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
#print("Creating ConjugateGradientSparse Object")

#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
#pres_lap = pl.pressure_laplacian(dim-1)
#name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
#A_sparse = sparse.load_npz(name_sparse_matrix)
#CG = cg.ConjugateGradientSparse(A_sparse)

#%%
input("Press Enter to continue...x1 s will be loaded")
d_name = "x_train_x_test_10000_1_10000_faulty_ritz_vectors_V2_for_3D_random_N"
filename = project_data_folder + d_name + str(dim-1) + "_tf"
with open(filename, "rb") as f:
    x_train1, x_test1 = pickle.load(f)
        
input("Press Enter to continue...x1s loaded. x2s will be loaded")


d_name = "x_train_x_test_10000_2_10000_faulty_ritz_vectors_V2_for_3D_random_N"
filename = project_data_folder + d_name + str(dim-1) + "_tf"
with open(filename, "rb") as f:
    x_train2, x_test2 = pickle.load(f)

input("Press Enter to continue...x2s are loaded. concat will happen")

x_train1 = tf.concat([x_train1, x_train2], 0)
del x_train2
gc.collect(generation=2)

input("Press Enter to continue...")

x_test1 = tf.concat([x_test1, x_test2], 0)

input("Press Enter to continue... saving will happen")
print(x_train1.shape)
print(x_test1.shape)

del x_test2
gc.collect(generation=2)

#%%

d_name = "x_train_x_test_20000_10000_faulty_ritz_vectors_V2_for_3D_random_N"
filename = project_data_folder + d_name + str(dim-1) + "_tf"
with open(filename, "wb") as f:
    pickle.dump([x_train1,x_test1], f)
    
#del x_train, x_test
    

















