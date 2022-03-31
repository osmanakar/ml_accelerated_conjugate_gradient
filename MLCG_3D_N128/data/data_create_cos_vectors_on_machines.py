#This part changed 
project_name = "MLCG_3D_N128"
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
dim = 128
dim2 = dim**3
#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
#pres_lap = pl.pressure_laplacian(dim-1)
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)
CG = cg.ConjugateGradientSparse(A_sparse)


#%%
#This part

num_ritz_vectors=10000
"""
from numba import njit, prange
@njit(parallel=True)
def create_cos_vec(dim2, num_ritz_vectors):
    p = dim2//num_ritz_vectors
    cos_vec = np.zeros(dim2,num_ritz_vectors)
    for j in prange(dim2):
        for i in range(num_ritz_vectors):
            cos_vec[j,i] = np.cos(np.pi*(dim2-i*p)*j/(dim2+1))            
    return cos_vec

cos_vec = create_cos_vec(dim2, num_ritz_vectors)
"""
def find_ijk(N,n):
    i = N//(n*n)
    j = (N-i*n*n)//n
    k = (N-i*n*n - n*j)
    return np.array([i,j,k])
#%%
p = dim2//num_ritz_vectors
print("p = ",p)
cos_vec = np.zeros([num_ritz_vectors,dim2])
n = dim

for j in range(num_ritz_vectors):
    if j%10 == 0:
        print(j)
    #[ji,jj,jk] = find_ijk(j,dim)
    N = j*p+1
    ji = N//(n*n)
    jj = (N-ji*n*n)//n
    jk = (N-ji*n*n - n*jj)
    for i in range(dim2):
        #[ii,ij,ik] = find_ijk(j,dim)
        N = i
        ii = N//(n*n)
        ij = (N-ii*n*n)//n
        ik = (N-ii*n*n - n*ij)
        cos_vec[j,i] = np.cos(np.pi*ii*(ji+0.5)/(dim))* np.cos(np.pi*ij*(jj+0.5)/dim)*np.cos(np.pi*ik*(jk+0.5)/dim)

#%%
A_cos_vec = np.zeros([10000,dim2])
for i in range(10000):
    A_cos_vec[i] = A_sparse.dot(cos_vec[i])
    A_cos_vec[i] = A_cos_vec[i]/np.linalg.norm(A_cos_vec[i])
    
ritz_values = CG.create_ritz_values(A_cos_vec)
sorted_ritz_vals = sorted(range(10000), key=lambda k: -ritz_values[k])
A_cos_vec = A_cos_vec[sorted_ritz_vals]
#%%
with open(project_data_folder+'A_cos_vec_10000_N'+str(dim-1)+'.npy', 'wb') as f:
    np.save(f, A_cos_vec)







