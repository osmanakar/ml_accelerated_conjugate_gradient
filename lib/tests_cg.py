#%%
import numpy as np
#lib_path = "C:/Users/osman/Desktop/research_teran/python/MLV1/remote/lib/"
lib_path = "path_to_the_folder_containing_library_functions"

import sys
sys.path.insert(1, lib_path)
import conjugate_gradient as cg
import pressure_laplacian as pl
import matplotlib.pyplot as plt

#%% ConjugateGradient cg_normal Test
A = np.array([[1,2,3,0],
              [2,3,4,10],
              [3,4,5,2],
              [0,10,2,8]])
CG =cg.ConjugateGradient(A)
x = np.array([1,1,2,10])
b = np.matmul(A,x)
x_init = np.zeros(4)
max_it = 100
tol = 1.0e-12
verbose=True
xx = CG.cg_normal(x_init,b,max_it,tol,verbose)

#%% Lanczos Iteration Test
Q, diag, sub_diag = CG.lanczos_iteration(b,3,tol)
print(np.matmul(Q,Q.transpose())) #this should be identity
print(np.matmul(np.matmul(Q,A),Q.transpose())) #this should be tridiagonal

#%% Appoximate Eigenvector pcg test:
pres_lap0 = pl.pressure_laplacian(64)
x_orig0 = np.random.rand(pres_lap0.m)
b0 = np.matmul(pres_lap.A, x_orig0)
b0 = b0/np.linalg.norm(b0)
CG0 = cg.ConjugateGradient(pres_lap.A)
num_modes = 16
max_it=100 
tol=1.0e-13
verbose=True
Q0 = CG0.create_approximate_eigenmodes(b0,num_modes)
lambda_vals = CG0.create_lambda_vals(Q0)
mult_precond = lambda x: CG0.mult_precond_approximate_eigenmodes(x,Q0,lambda_vals)
x0, res_arr0 = CG0.pcg_normal(np.zeros(pres_lap0.m),b0,mult_precond,max_it,tol,verbose)

%matplotlib qt
plt.plot(np.log10(res_arr0))

#%% Restarted PCG Automatic Test
pres_lap0 = pl.pressure_laplacian(64)
x_orig1 = np.random.rand(pres_lap0.m)
b1 = np.matmul(pres_lap.A, x_orig1)
b1 = b1/np.linalg.norm(b1)
num_modes = 16
max_outer_it = 100
pcg_inner_it = 1
tol = 1.0e-13,
method = "approximate_eigenmodes"
num_modes = 16
verbose = True
x1, res_arr1= CG0.restarted_pcg_automatic(b1, max_outer_it, pcg_inner_it, tol, method , num_modes , verbose)

plt.plot(np.log10(res_arr1))

#%% Restarted PCG Manuel Test
pres_lap0 = pl.pressure_laplacian(64)
x_orig2 = np.random.rand(pres_lap0.m)
b2 = np.matmul(pres_lap.A, x_orig2)
b2 = b2/np.linalg.norm(b2)
num_modes = 16
max_outer_it = 10
pcg_inner_it = 1
tol = 1.0e-13,
method = "approximate_eigenmodes"
num_modes = 16
verbose = True

def mult_precond_method(CG_, x, b):
    Q = CG_.create_approximate_eigenmodes(b,num_modes)
    lambda_ =  CG_.create_lambda_vals(Q)
    return CG_.mult_precond_approximate_eigenmodes(x,Q,lambda_)
    
x2, res_arr2= CG0.restarted_pcg_manual(b2, mult_precond_method,max_outer_it, pcg_inner_it, tol, verbose)

plt.plot(np.log10(res_arr2))

























