project_name = "MLCG_3D_N128"
#project_folder_subname = "MLV6_T1_N511_ritz_vectors_10000_2500"
#print(project_folder_subname)
#project_folder_general = "/home/osman/projects/ML_preconditioner_project/"+project_name+"/"
project_folder_general = "/home/oak/projects/ML_preconditioner_project/"+project_name+"/"
project_data_folder = "/home/oak/projects/ML_preconditioner_project/data/"+project_name+"/"

import sys
sys.path.insert(1, project_folder_general+'../lib/')
import numpy as np
#from tensorflow import keras
#from tensorflow.keras import layers
#import os
#import tensorflow as tf 
#import gc
import scipy.sparse as sparse
#import matplotlib.pyplot as plt

#import conjugate_gradient as cg
import pressure_laplacian as pl
#import helper_functions as hf


#%% Creating ConjugateGradientSparse Object
print("Creating 3D poisson Matrix Object")
dim = 256
dim2 = dim**3


pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
sparse.save_npz(name_sparse_matrix, pres_lap.A_sparse)
#A_sparse = sparse.load_npz(name_sparse_matrix)
#CG = cg.ConjugateGradientSparse(pres_lap.A_sparse)
#CG = cg.ConjugateGradientSparse(A_sparse)
#%%
dim = 16
dim2 = dim**3
pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
#A_sparse_test1 = pres_lap.A_sparse.copy()
A_sparse_test2 = pres_lap.A_sparse.copy()

#%%
AD1 = A_sparse_test1.toarray()
AD2 = A_sparse_test2.toarray()


