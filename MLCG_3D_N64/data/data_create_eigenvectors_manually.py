import numpy as np
from numpy import cos,sin
pi = np.pi
import os
#project_name = "PFI_MLV1"
project_name = "MLCG_3D_N64"
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
#import matplotlib.pyplot as plt
import scipy.sparse as sparse

dim = 8
dim2 = dim**3



#%% 3D ritz vector creation
#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = pres_lap.A_sparse
#A_dense = A_sparse.toarray()
#%%
name_sparse_matrix = dir_path+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
#pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = sparse.load_npz(name_sparse_matrix)

dangle = pi/(dim+1)
V = np.zeros([dim2,dim2])
for i in range(dim):
    b = -1.0
    a = sin(dangle*i)/(5-cos(dangle*i))
    for j in range(dim):
        d = -1.0
        c = sin(dangle*j)/(5-cos(dangle*j))
        for k in range(dim):
            f = -1.0
            e = sin(dangle*k)/(5-cos(dangle*k))
            
            n = i*dim*dim + j*dim + k
            for i0 in range(dim):
                for j0 in range(dim):
                    for k0 in range(dim):
                        m = i0*dim*dim + j0*dim + k0
                        V[n,m] = (a*cos(i*i0*dangle)+b*sin(i*i0*dangle))*(a*cos(i*i0*dangle)+b*sin(i*i0*dangle))*(a*cos(i*i0*dangle)+b*sin(i*i0*dangle))
            
#%%

def get_3D_eigenvector(dim,i,j,k): 
    ii = 2*i-1
    jj = 2*j-1
    kk = 2*k-1
    dangle = pi/(dim+1)
    eigvec = np.zeros([dim**3])
    b = -1.0
    a = sin(dangle*ii)/(5-cos(dangle*ii))  
    d = -1.0
    c = sin(dangle*jj)/(5-cos(dangle*jj))
    f = -1.0
    e = sin(dangle*kk)/(5-cos(dangle*kk))
    for i0 in range(dim):
        for j0 in range(dim):
            for k0 in range(dim):
                m = i0*dim*dim + j0*dim + k0
                eigvec[m] = (a*cos(ii*i0*dangle)+b*sin(ii*i0*dangle))*(c*cos(jj*j0*dangle)+d*sin(jj*j0*dangle))*(e*cos(kk*k0*dangle)+f*sin(kk*k0*dangle))
    return eigvec/np.linalg.norm(eigvec)
    
    
#%%
i=2
j=2
k=2
eigvec = get_3D_eigenvector(dim,i,j,k)
Aeigvec = A_sparse.dot(eigvec)
eigval = np.dot(eigvec, Aeigvec)
print(0, np.linalg.norm(eigval*eigvec - Aeigvec))










    
    
    