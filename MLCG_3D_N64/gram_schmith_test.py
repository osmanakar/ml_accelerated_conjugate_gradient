
import numpy as np
import os
#project_name = "PFI_MLV1"
project_name = "MLCG_3D_N32"
#project_folder_general = "/home/osman/projects/ML_preconditioner_project/"+project_name+"/"

dir_path = os.path.dirname(os.path.realpath(__file__))+'/'
lib_path = os.path.dirname(os.path.realpath(__file__))+'/../lib/'
#%%
import sys
sys.path.insert(1, lib_path)
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse as sparse

#%% 3D ritz vector creation
dim = 32
dim2 = dim**3
#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = pres_lap.A_sparse
A_dense = A_sparse.toarray()

#%% Reading via different method
matrix_filename = dir_path+"../MLCG_3D_N32/data/output32_3d"

#%%
rand_vec_x = np.random.normal(0,1, [dim2])
b = A_sparse.dot(rand_vec_x)
#%%
data_folder_name = dir_path+"data/gram_schmith_tests/output3D_"+str(dim-1)
n = 10
b = hf.get_frame_from_source(n, data_folder_name)

#%%
tol = 1.0e-4
Q = A_dense.copy()
for j in range(dim2):
    if j%50 == 0:
        print(j)
    Qj_norm = np.linalg.norm(Q[j])
    if Qj_norm > tol:
        qj = Q[j]/Qj_norm
        for k in range(j+1,dim2):
            Q[k] = Q[k] - np.dot(qj,Q[k])*qj
            
#%%
for j in range(dim2):
    Qj_norm = np.linalg.norm(Q[j])
    if Qj_norm > tol:
        Q[j] = Q[j]/Qj_norm
#%%
b_init = np.zeros([dim2])

for i in range(dim2):
    b_init = b_init + Q[i]*np.dot(Q[i],b)
    
    