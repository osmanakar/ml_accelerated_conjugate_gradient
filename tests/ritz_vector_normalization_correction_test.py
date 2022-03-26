
import numpy as np
#project_name = "PFI_MLV1"
project_name = "MLCG_3D_N32"
#old places 
#project_folder_general = "/home/osman/projects/ML_preconditioner_project/"+project_name+"/"
apple = "/Users/osmanakar/"
windows = "C:/Users/osman/"
project_folder_general = apple + "OneDrive/research_teran/python/"+project_name+"/"

#%%
from numpy.linalg import norm
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import sys
import os
#sys.path.insert(1, lib_path)
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path+'/../../lib/')


import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf


#%% 3D ritz vector creation
dim = 10
dim2 = dim**3
#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = pres_lap.A_sparse

#%%
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)

#%%
CG = cg.ConjugateGradientSparse(A_sparse)

#%% Creating random vector to start lanczos iteration -- Type1
#b = hf.get_frame(100,dim,'mac')
rand_vec = np.random.normal(0,1, [dim2])
#rand_vec = CG.multiply_A_sparse(rand_vec_x)
zero_idx = pres_lap.zero_indexes()
one_vec = np.ones([dim2])
one_vec[zero_idx] = 0
rand_vec[zero_idx] = 0
sum_rand_vec = sum(rand_vec)
rand_vec = rand_vec - one_vec*sum_rand_vec/(dim2-len(zero_idx))

#%% Test this created rand_vec is indeed in the Image of A
x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(rand_vec.shape),rand_vec,1000,1e-12,True)
#%matplotlib qt

#%%
rand_vec_x = np.random.normal(0,1, [dim2])
rand_vec = A_sparse.dot(rand_vec_x)
#%%
num_vectors = 100
W, diagonal, sub_diagonal = CG.lanczos_iteration(rand_vec, num_vectors, 1.0e-10)

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
A_dense = A_sparse.toarray()
WTAW = np.matmul(W,np.matmul(A_dense,W.transpose()))
print(0,np.linalg.norm(WTAW-tri_diag))

#%%%
#test_vectors = eigenvectors_A[1:10]
test_vectors = W[0:10].copy()
tt = np.matmul(test_vectors,test_vectors.transpose())
tat = np.matmul(test_vectors, np.matmul(A_dense,test_vectors.transpose())) 

#%%
eigvals,Q0 = np.linalg.eigh(tri_diag)
eigvals = np.real(eigvals)
Q0 = np.real(Q0)
ritz_vectors = np.matmul(W.transpose(),Q0).transpose()
sorted_eig_vals = sorted(range(num_vectors), key=lambda k: -eigvals[k])
ritz_vectors = ritz_vectors[sorted_eig_vals]
ritz_vals = eigvals[sorted_eig_vals]

#%%
t1 = np.matmul(Q0.transpose(),np.matmul(tri_diag,Q0))

#%%%
#test_vectors = eigenvectors_A[1:10]
test_vectors = W[0:30].copy()
tt = np.matmul(test_vectors,test_vectors.transpose())
tat = np.matmul(test_vectors, np.matmul(A,test_vectors.transpose())) 


#%%
i = 0
j = 2
print(norm(ritz_vectors[i]-ritz_vectors[j]))
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[j])))
print(CG.dot(ritz_vectors[i], CG.multiply_A_sparse(ritz_vectors[i])))
print(np.dot(ritz_vectors[i], np.matmul(A,ritz_vectors[i])))
print(ritz_vals[i])

#%%
print(ritz_vals[0:20])






#%% 2D test for ritz vector creation
dim = 16
dim2 = dim**2
pres_lap = pl.pressure_laplacian_sparse(dim-1)
A_sparse = pres_lap.A_sparse
CG = cg.ConjugateGradientSparse(A_sparse)

A_dense = A_sparse.toarray()
#%% Create matrix to start with
rand_vec_x = np.random.normal(0,1, [dim2])
rand_vec = A_sparse.dot(rand_vec_x)
#%%
num_vectors = 30
W, diagonal, sub_diagonal = CG.lanczos_iteration(rand_vec, num_vectors, 1.0e-10)
tri_diag = np.zeros([num_vectors,num_vectors])
for i in range(1,num_vectors-1):
    tri_diag[i,i] = diagonal[i]
    tri_diag[i,i+1] = sub_diagonal[i]
    tri_diag[i,i-1]= sub_diagonal[i-1]
tri_diag[0,0]=diagonal[0]
tri_diag[0,1]=sub_diagonal[0]
tri_diag[num_vectors-1,num_vectors-1]=diagonal[num_vectors-1]
tri_diag[num_vectors-1,num_vectors-2]=sub_diagonal[num_vectors-2]
#%% This part checks if tridiagonalization worked.
WTAW = np.matmul(W,np.matmul(A_dense,W.transpose()))
print(0,np.linalg.norm(WTAW-tri_diag))
#%%
eigvals,Q0 = np.linalg.eigh(tri_diag)
eigvals = np.real(eigvals)
Q0 = np.real(Q0)
ritz_vectors = np.matmul(W.transpose(),Q0).transpose()
sorted_eig_vals = sorted(range(num_vectors), key=lambda k: -eigvals[k])
ritz_vectors = ritz_vectors[sorted_eig_vals]
ritz_vals = eigvals[sorted_eig_vals]

#%%
t1 = np.matmul(Q0.transpose(),np.matmul(tri_diag,Q0))

#%%%
#test_vectors = eigenvectors_A[1:10]
test_vectors = W[0:30].copy()
tt = np.matmul(test_vectors,test_vectors.transpose())
tat = np.matmul(test_vectors, np.matmul(A_dense,test_vectors.transpose())) 






#%% A_dense Eigenvector tests
A = A_sparse.toarray()
print("creating eigenvalues")
eigvals_A, eigenvectors_A = np.linalg.eigh(A)
idx_ = eigvals_A.argsort()[::-1]
eigvals_A = np.real(eigvals_A[idx_])
eigenvectors_A = np.real(eigenvectors_A[:,idx_].transpose())

#%% Testing eigenvectors
k=1
print(0, norm(eigenvectors_A[k]*eigvals_A[k] - np.matmul(A,eigenvectors_A[k])))
i = 1
j = 3
print(0, np.dot(eigenvectors_A[i],eigenvectors_A[j]))
print(2, norm(eigenvectors_A[i]+eigenvectors_A[j])**2)
print(0, CG.dot(eigenvectors_A[i], CG.multiply_A_sparse(eigenvectors_A[j])))
print(eigvals_A[i], CG.dot(eigenvectors_A[i], CG.multiply_A_sparse(eigenvectors_A[i])))
#print(np.dot(eigenvectors_A[i], np.matmul(A,eigenvectors_A[i])))

#%matplotlib qt
dist_b = np.matmul(eigenvectors_A,x_sol_cg)
plt.bar(list(range(dim2,0,-1)),dist_b)





#%%
ritz_values = CG.create_ritz_values(ritz_vectors)
idx_ = ritz_values.argsort()[::-1]
ritz_vectors = ritz_vectors[idx_]
#%%
import gc
CG.A=None
#pres_lap.A=None
#del pres_lap
gc.collect(generation=2)

#%%
dist_b = np.matmul(ritz_vectors, b)
plt.bar(list(range(num_ritz_vectors,0,-1)),dist_b)

#%%
print(np.where(ritz_values<1e-10))
#%%
num_ritz_vectors=2500
cut_idx = int(num_ritz_vectors/2)-200
num_zero_ritz_vals = 1
sample_size = 10000
coef_matrix = np.random.normal(0,1, [num_ritz_vectors-num_zero_ritz_vals,sample_size])
coef_matrix[cut_idx:num_ritz_vectors-num_zero_ritz_vals] = 9*np.random.normal(0,1, [num_ritz_vectors-num_zero_ritz_vals-cut_idx,sample_size])
#%%
b_rhs = np.zeros([sample_size,dim2])

#%%

for it in range(10):
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
    #print(norm(b_rhs)**2)
#%%
dist_b = np.matmul(ritz_vectors,b_rhs[0])
plt.bar(list(range(1000,0,-1)),dist_b)
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