project_name = "MLCG_3D_N64"
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
import os
#import tensorflow as tf 
import gc
import scipy.sparse as sparse
#import matplotlib.pyplot as plt

import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf
#from tensorflow.keras import backend as K


#%% Creating ConjugateGradientSparse Object
print("Creating 3D poisson Matrix Object")
dim = 32
dim2 = dim**3


#pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)
#CG = cg.ConjugateGradientSparse(pres_lap.A_sparse)
CG = cg.ConjugateGradientSparse(A_sparse)

#%% Creating Ritz vectors...
num_ritz_vectors = 10000
#b = hf.get_frame(100,dim,'mac')
rand_vec_x = np.random.normal(0,1, [dim2])
rand_vec = CG.multiply_A_sparse(rand_vec_x)
#ritz_vectors = CG.create_ritz_vectors(rand_vec, num_ritz_vectors)
#%%
print("Creating Ritz Vectors")
num_vectors = num_ritz_vectors
#W, diagonal, sub_diagonal = CG.lanczos_iteration(rand_vec, num_vectors, 1.0e-12)
W, diagonal, sub_diagonal = CG.lanczos_iteration_with_normalization_correction(rand_vec, num_vectors, 1.0e-10)

#%%
tri_diag = np.zeros([num_vectors,num_vectors])
for i in range(1,num_vectors-1):
    tri_diag[i,i] = diagonal[i]
    tri_diag[i,i+1] = sub_diagonal[i]
    tri_diag[i,i-1]= sub_diagonal[i-1]
tri_diag[0,0]=diagonal[0]
tri_diag[0,1]=sub_diagonal[0]
tri_diag[num_vectors-1,num_vectors-1]=diagonal[num_vectors-1]
tri_diag[num_vectors-1,num_vectors-2]=sub_diagonal[num_vectors-2]

#%%
print("Testing ritz vectors...")
print("Converting sparse matrix into dense matrix...")
A_dense = A_sparse.toarray()
print("Calculating W^T * A * W via np.matmul...")
WTAW = np.matmul(W,np.matmul(A_dense,W.transpose()))
print("err norm = ",np.linalg.norm(WTAW-tri_diag))


#%% Calculating eigenvectors of the tridiagonal matrix
print("Calculating eigenvectors of the tridiagonal matrix")
eigvals, Q0 = np.linalg.eigh(tri_diag)
eigvals = np.real(eigvals)
#%%
Q0 = np.real(Q0)
#%%
ritz_vectors = np.zeros(W.shape)
#%%
for i in range(10):
    ll = 1000*i
    rr = ll+1000
    print(ll)
    #% %
    ritz_vectors[ll:rr] = np.matmul(W.transpose(),Q0[:,ll:rr]).transpose()
#%%
#Q = np.zeros([num_vectors,CG.n])
sorted_eig_vals = sorted(range(num_vectors), key=lambda k: -eigvals[k])
#%%
del Q0
del W
import gc
#pres_lap.A=None
#del pres_lap
gc.collect(generation=2)
#%%
print("Ritz vectors, reordering")
ritz_vectors = ritz_vectors[sorted_eig_vals]
#%%
ritz_values = eigvals[sorted_eig_vals]

#%% Testing
print("testing")
i = 600
j = 600
print("i = ",i,", j = ",j)
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[j])))
print(ritz_values[i])
i = 600
j = 900
print("i = ",i,", j = ",j)
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[j])))
print(ritz_values[i])

#%% Sacving and loading ritz values
print("Saving Ritz Values")
#MLproject_data_folder = "/media/data/osman/ML_preconditioner_project/data"
with open(project_data_folder+'ritz_vectors_'+str(num_ritz_vectors)+'_3D_N'+str(dim-1)+'.npy', 'wb') as f:
    np.save(f, ritz_vectors)
    
"""
MLproject_data_folder = project_folder_general+"data/"
with open(MLproject_data_folder+'ritz_vectors_10000_N'+str(dim-1)+'.npy', 'rb') as f:
    ritz_vectors = np.load(f)
"""
print("Ritz Values")
print("ritz_values[0:20]", ritz_values[0:20])
print("ritz_values[-20:-1]", ritz_values[num_ritz_vectors-20:num_ritz_vectors])

#%%
find_nonzero_ritz_value = True
tol = 1.0e-6
num_zero_ritz_vals = 0
while find_nonzero_ritz_value:
    if ritz_values[num_ritz_vectors - 1 - num_zero_ritz_vals] < tol:
        num_zero_ritz_vals = num_zero_ritz_vals + 1
    else:
        find_nonzero_ritz_value = False

print("num_zero_ritz_vals = ", num_zero_ritz_vals)
print("ritz_vals[idx-5,idx+5] = ", ritz_values[num_ritz_vectors - 1 - num_zero_ritz_vals-5:num_ritz_vectors - 1 - num_zero_ritz_vals+5])


"""
#%%
for i in range(num_ritz_vectors):
    if i%100==0:
        print(i)
    if abs(sum(ritz_vectors[i]))>1.0e-9:
        print(i, sum(ritz_vectors[i]))
"""

#%%
#x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(ritz_vectors[181].shape),ritz_vectors[181],1000,1e-14,True)
#%%
#plt.plot(np.log10(res_arr_cg))
#%%
#ritz_values = CG.create_ritz_values(ritz_vectors)

"""
#%%
num_ritz_vectors=10000
cut_idx = int(num_ritz_vectors/2)-500
num_zero_ritz_vals = 5
sample_size = 10000
coef_matrix = np.random.normal(0,1, [num_ritz_vectors-num_zero_ritz_vals,sample_size])
coef_matrix[cut_idx:num_ritz_vectors-num_zero_ritz_vals] = 9*np.random.normal(0,1, [num_ritz_vectors-num_zero_ritz_vals-cut_idx,sample_size])
#%%
b_rhs = np.zeros([sample_size,dim2])

#%% 

for it in range(0,10):
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
    #%
    print(norm(b_rhs)**2)

#%%
#import gc
#del ritz_vectors
del b_rhs
gc.collect(generation=2)

#%%
dist_b = np.matmul(ritz_vectors,b0)
plt.bar(list(range(10000,0,-1)),dist_b)
#%%
#dist_b = np.matmul(eigenvectors_A,b_rhs[10])
dist_b = np.matmul(eigenvectors_A,b)
plt.bar(list(range(dim2,0,-1)),dist_b)

#%% Create direcly from eigenvectors
eigvals_A, eigenvectors_A = np.linalg.eigh(pres_lap.A)
idx_ = eigvals_A.argsort()[::-1]
#%%
eigvals_A = np.real(eigvals_A[idx_])
eigenvectors_A = np.real(eigenvectors_A[:,idx_].transpose())
#%% test
k=0
print(np.linalg.norm(eigenvectors_A[k]*eigvals_A[k] - np.matmul(pres_lap.A,eigenvectors_A[k])))    
#%%
cut_idx = 1800
coef_matrix = np.random.normal(0,1, [dim2-5,sample_size])
coef_matrix[cut_idx:dim2-5] = 9*np.random.normal(0,1, [dim2-5-cut_idx,sample_size])
#%%
b_rhs = np.matmul(eigenvectors_A[0:dim2-5].transpose(),coef_matrix).transpose()

#%%
coef_matrix2 = np.random.normal(0,1, [dim2-5,1])
b_rhs2 = np.matmul(eigenvectors_A[0:dim2-5].transpose(),coef_matrix2).transpose()


#%% Test 
m1 = 10
m2 = 100
print(sum(sum(b_rhs)), np.linalg.norm(b_rhs)**2, sum(b_rhs[m1]), np.linalg.norm(b_rhs[m1]),sum(b_rhs[m2]), np.linalg.norm(b_rhs[m2]))

#%% Create random from low rank
rank = 500
coef_matrix = np.random.normal(0,1, [dim2,rank])
base = np.matmul(eigenvectors_A.transpose(),coef_matrix)
#%%
coef_matrix2 = np.random.normal(0,1, [rank,sample_size])
b_rhs = np.matmul() ...

#%%
MLproject_data_folder = project_folder_general+"data/"
with open(MLproject_data_folder+'b_rhs_10000_2_10000_ritz_vectors_first_half_10_last_half_90_random_N'+str(dim-1)+'.npy', 'wb') as f:
    np.save(f, b_rhs)
#%%
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
ritz_values = CG.create_ritz_values(ritz_vectors)
#%%
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(list(range(sample_size)) , test_size=0.1, train_size = 0.9)

with open(MLproject_data_folder+'train_idx_20000'+'.npy', 'wb') as f:
    np.save(f, train_idx)
with open(MLproject_data_folder+'test_idx_20000'+'.npy', 'wb') as f:
    np.save(f, test_idx)

#%%
"""