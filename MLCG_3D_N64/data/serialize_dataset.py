#This part changed 
project_name = "MLCG_3D_N64"
#print(project_folder_subname)
#project_folder_general = "/home/osman/projects/ML_preconditioner_project/"+project_name+"/"
project_folder_general = "/home/oak/projects/ML_preconditioner_project/"+project_name+"/"
project_data_folder = "/home/oak/projects/ML_preconditioner_project/data/"+project_name+"/"


import sys
import numpy as np
import os
import gc
import scipy.sparse as sparse
#import matplotlib.pyplot as plt
from numpy.linalg import norm

sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf



#%% Creating ConjugateGradientSparse Object
print("Creating ConjugateGradientSparse Object")
dim = 64
dim2 = dim**3
#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
#pres_lap = pl.pressure_laplacian(dim-1)
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)
CG = cg.ConjugateGradientSparse(A_sparse)



#%%
#This part 
with open(project_data_folder+'b_rhs_20000_10000_faulty_ritz_vectors_V2_for_3D_random_N'+str(dim-1)+'.npy', 'rb') as f:
    ritz_vectors = np.load(f)

num_vectors = 20000
foldername = project_data_folder+'b_rhs_20000_10000_faulty_ritz_vectors_V2_for_3D_random_N'+str(dim-1)+'/'

for i in range(num_vectors):
    if i%100 == 0:
        print(i)
    with open(foldername+str(i)+'.npy', 'wb') as f:
        Ar = A_sparse.dot(ritz_vectors[i])
        Ar = Ar/np.linalg.norm(Ar)
        np.save(f, np.array(Ar,dtype=np.float32))
        #np.save(f, ritz_vectors[i])




















