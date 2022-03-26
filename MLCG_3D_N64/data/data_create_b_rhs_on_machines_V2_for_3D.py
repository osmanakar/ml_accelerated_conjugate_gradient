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


#%% Saving and loading ritz values
print("Loading Ritz Vectors")
with open(project_data_folder+'ritz_vectors_10000_3D_N'+str(dim-1)+'.npy', 'rb') as f:
    ritz_vectors = np.load(f)
    
#ritz_vectors = ritz_vectors[::-1]
print(sum(sum(ritz_vectors[0:10000-100])))
print("Computing Ritz Values")
ritz_values = CG.create_ritz_values(ritz_vectors)
print(ritz_values[9950:10000])

#%%
for i in range(1000):
    if abs(sum(ritz_vectors[i])) >1.0e-5:
        print(i, sum(ritz_vectors[i]))

#%% Testing
print("testing")
i = 6000
j = 6000
print("i = ",i,", j = ",j)
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[j])))
print(ritz_values[i])
i = 6000
j = 9000
print("i = ",i,", j = ",j)
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[j])))
print(ritz_values[i])
#%%

print("Ritz Values")
print(ritz_values[0:20])
print(ritz_values[9980:10000])

#%%
#This part
num_ritz_vectors=10000
cut_idx = int(num_ritz_vectors/2)-500
num_zero_ritz_vals = 1
sample_size = 20000
coef_matrix = np.random.normal(0,1, [num_ritz_vectors-num_zero_ritz_vals,sample_size])
coef_matrix[cut_idx:num_ritz_vectors-num_zero_ritz_vals] = 9*np.random.normal(0,1, [num_ritz_vectors-num_zero_ritz_vals-cut_idx,sample_size])
print("norm(coef_matrix = ", np.linalg.norm(coef_matrix))
print("coef_matrix[0,0]", coef_matrix[0,0])
print("coef_matrix[10,10]", coef_matrix[10,10])

#%%
#This part
num_ritz_vectors=10000
num_zero_ritz_vals = 1
cut_idx0 = 0
cut_idx1 = int(num_ritz_vectors/8)
cut_idx2 = int(num_ritz_vectors*5/16)
cut_idx3 = int(num_ritz_vectors*10/16)
cut_idx4 = num_ritz_vectors-num_zero_ritz_vals
#%% Reversed cut indices:
r_cut_idx0 = cut_idx4 - cut_idx0
r_cut_idx1 = cut_idx4 - cut_idx1
r_cut_idx2 = cut_idx4 - cut_idx2
r_cut_idx3 = cut_idx4 - cut_idx3
r_cut_idx4 = 0

sample_size = 20000
coef_matrix = np.random.normal(0,1, [cut_idx4,sample_size])
coef_matrix[r_cut_idx1:r_cut_idx0] = 9*np.random.normal(0,1, [r_cut_idx0-r_cut_idx1,sample_size])
coef_matrix[r_cut_idx3:r_cut_idx2] = 9*np.random.normal(0,1, [r_cut_idx2-r_cut_idx3,sample_size])

#%%
b_rhs = np.zeros([sample_size,dim2])

#%% 
for it in range(0,20):
    small_size = 1000
    #it = 6
    l_b = small_size*it
    r_b = small_size*(it+1)
    print(it)
    b_rhs[l_b:r_b] = np.matmul(ritz_vectors[0:num_ritz_vectors-num_zero_ritz_vals].transpose(),coef_matrix[:,l_b:r_b]).transpose()
    
    #% % Making sure b is in the range of A
    for i in range(l_b,r_b):
        if i%100 == 0:
            print(i)
        #b_rhs[0] = b_rhs[0] - sum(b_rhs[0][reduced_idx])/len(reduced_idx)
        #b_rhs[0][zero_idxs]=0
        b_rhs[i]=b_rhs[i]/np.linalg.norm(b_rhs[i])
    print(norm(b_rhs)**2)

#%%
#import gc
del ritz_vectors
#del b_rhs
gc.collect(generation=2)

#%% Test 
m1 = 10
m2 = 100
print(m1,m2)
print(sum(sum(b_rhs)), np.linalg.norm(b_rhs)**2, sum(b_rhs[m1]), np.linalg.norm(b_rhs[m1]),sum(b_rhs[m2]), np.linalg.norm(b_rhs[m2]))

#%%
with open(project_data_folder+'b_rhs_20000_10000_faulty_ritz_vectors_V2_for_3D_random_N'+str(dim-1)+'.npy', 'wb') as f:
    np.save(f, b_rhs)
#%%
"""
MLproject_data_folder = project_folder_general+"data/"
with open(MLproject_data_folder+'b_rhs_20000_2500_ritz_vectors_first_half_10_last_half_90_random_N'+str(dim-1)+'.npy', 'rb') as f:
    b_rhs = np.load(f)

#%% Sacving and loading ritz values
MLproject_data_folder = project_folder_general+"data/"
with open(MLproject_data_folder+'ritz_vectors_10000_N'+str(dim-1)+'.npy', 'wb') as f:
    np.save(f, ritz_vectors)
#%%
MLproject_data_folder = project_folder_general+"data/"
with open(MLproject_data_folder+'ritz_vectors_10000_N'+str(dim-1)+'.npy', 'rb') as f:
    ritz_vectors = np.load(f)
#%%
"""




