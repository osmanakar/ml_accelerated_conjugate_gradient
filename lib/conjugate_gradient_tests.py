import numpy as np
import sys
import os
import scipy.sparse as sparse

#lib_path = "C:/Users/osman/OneDrive/research_teran/python/lib/"
#lib_path = "/Users/osmanakar/OneDrive/research_teran/python/lib/"
#lib_path = "path_to_the_folder_containing_library_functions"
project_folder_general = os.path.dirname(os.path.realpath(__file__))+"/"
#%%
sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import matplotlib.pyplot as plt
import helper_functions as hf


#%% ConjugateGradient cg_normal Test
A = np.array([[1,2,3,0],
              [2,3,4,10],
              [3,4,5,2],
              [0,10,2,8]])
CG =cg.ConjugateGradient(A)
x = np.array([1,1,2,10])
b = np.matmul(A,x)
x_init = np.zeros(4)
max_it = 4
tol_cg = 1.0e-12
verbose=True
x,res_arr_cg  = CG.cg_normal(x_init,b,max_it,tol_cg,verbose)

#%% Spectral Preconditoner pcg test:
pres_lap = pl.pressure_laplacian(64)
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)
CG = cg.ConjugateGradient(pres_lap.A)
num_vectors = 16
max_it=100
tol_pcg=1.0e-13
verbose=True
x_init=np.zeros(pres_lap.m)
Q = CG.create_ritz_vectors(b,num_vectors)
ritz_vals = CG.create_ritz_values(Q)
mult_precond = lambda x: CG.mult_precond_method1(x,Q,ritz_vals)
x, res_arr_pcg = CG.pcg_normal(x_init,b,mult_precond,max_it,tol_pcg,verbose)
#%%
%matplotlib qt
plt.plot(np.log10(res_arr_pcg))


#%% ConjugateGradient lanczos_pcg Test (cpp version). Not recommended to use.
A = np.array([[1,2,3,0],
              [2,3,4,10],
              [3,4,5,2],
              [0,10,2,8]])
CG =cg.ConjugateGradient(A)
x = np.array([1,1,2,10])
b = np.matmul(A,x)
x_init = np.zeros(4)
max_it = 5
tol = 1.0e-12
verbose=True
M = np.array([[1,0,0,0],
              [0,1/3,0,0],
              [0,0,1/5,0],
              [0,0,0,1/8]])
mult_precond = lambda x : np.matmul(M,x)
xx, res_arr = CG.lanczos_pcg_old(x_init,b,mult_precond, max_it,tol,verbose)

#%% Lanczos Iteration Test
tol = 1.0e-12
x = np.array([1,1,2,10])
b = np.matmul(A,x)
A = np.array([[1,2,3,0],
              [2,3,4,10],
              [3,4,5,2],
              [0,10,2,8]])
CG =cg.ConjugateGradient(A)
lanczos_vectors, diag, sub_diag = CG.lanczos_iteration(b,3,tol)
print("this must be identity:")
print(np.matmul(lanczos_vectors,lanczos_vectors.transpose())) #this should be identity
print("this must be tridiagonal:")
print(np.matmul(np.matmul(lanczos_vectors,A),lanczos_vectors.transpose())) #this should be tridiagonal



#%% Restarted PCG Manuel Test
pres_lap = pl.pressure_laplacian(64)
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)
num_vectors = 16
max_outer_it = 10
pcg_inner_it = 1
tol = 1.0e-13,
verbose = True

def mult_precond_method(CG_, x, b):
    Q = CG_.create_ritz_vectors(b,num_vectors)
    lambda_ =  CG_.create_ritz_values(Q)
    return CG_.mult_precond_method1(x,Q,lambda_)
    
x, res_arr= CG.restarted_pcg_manual(b, mult_precond_method, max_outer_it, pcg_inner_it, tol, verbose)
plt.plot(np.log10(res_arr))

#%% Jacobi Iteration Test
pres_lap = pl.pressure_laplacian(64)
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)
CG = cg.ConjugateGradient(pres_lap.A)
num_vectors = 16
Q = CG.create_ritz_vectors(b,num_vectors)

noise_strength = 4
noise_variance = 10
Q_noise = np.random.normal(0,noise_variance, Q.shape)
Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
Q_noisy = Q+Q_noise

A_tilde = np.matmul(Q_noisy,Q_noisy.transpose())

threshold = 1e-10
max_it = 10000
verbose = True
R = CG.jacobi_diagonalization(A_tilde, threshold, max_it, verbose)
D = np.matmul(R.transpose(),np.matmul(A_tilde,R))

#%%
max_it=100
tol=1.0e-13
verbose=True
ritz_vals = CG.create_ritz_values(Q)
mult_precond = lambda x: CG.mult_precond_method1(x,Q,ritz_vals)
x, res_arr = CG.pcg_normal(np.zeros(pres_lap.m),b,mult_precond,max_it,tol,verbose)

#%% Gauss Seidel Test
A_dense = np.array([[10,2,3,0],
              [2,20,4,10],
              [3,4,50,2],
              [0,10,2,80]])

A = sparse.csr_matrix(A_dense)
CG = cg.ConjugateGradientSparse(A)
x_exact = np.random.rand(4)
b = A.dot(x_exact)
x_init = np.zeros([4])
tol = 1.0e-12
max_it = 100
verbose = True
CG.create_lower_and_upper_matrices()  
x, res_arr = CG.gauss_seidel_sparse(b, x_init,max_it,tol,verbose)
















