import numpy as np
lib_path = "C:/Users/osman/OneDrive/research_teran/python/lib/"
#lib_path = "path_to_the_folder_containing_library_functions"

import sys
sys.path.insert(1, lib_path)
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
tol = 1.0e-12
verbose=True
xx = CG.cg_normal(x_init,b,max_it,tol,verbose)

#%% ConjugateGradient lanczos_pcg Test
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

#%% Spectral Preconditoner pcg test:
pres_lap = pl.pressure_laplacian(64)
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)
CG = cg.ConjugateGradient(pres_lap.A)
num_vectors = 16
max_it=100
tol=1.0e-13
verbose=True
Q = CG.create_ritz_vectors(b,num_vectors)
ritz_vals = CG.create_ritz_values(Q)
mult_precond = lambda x: CG.mult_precond_method1(x,Q,ritz_vals)
x, res_arr = CG.pcg_normal(np.zeros(pres_lap.m),b,mult_precond,max_it,tol,verbose)
#%%
%matplotlib qt
plt.plot(np.log10(res_arr))

#%% Restarted PCG Automatic Test --- Redo this
pres_lap = pl.pressure_laplacian(64)
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)
num_vectors = 16
max_outer_it = 100
pcg_inner_it = 1
tol = 1.0e-13,
method = "approximate_eigenmodes"
num_modes = 16
verbose = True
x1, res_arr1= CG0.restarted_pcg_automatic(b1, max_outer_it, pcg_inner_it, tol, method , num_modes , verbose)

plt.plot(np.log10(res_arr1))

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

#%% Restarted PCG Manuel Test with Noisy Vectors

pres_lap = pl.pressure_laplacian(64)
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)
num_vectors = 16
max_outer_it = 10
pcg_inner_it = 1
tol = 1.0e-13,
verbose = True

def mult_precond_method_noisy_ritz_vecs_projected_onto_lanczos(CG_, x, b):
    Q = CG_.create_ritz_vectors(b,num_vectors)
    noise_strength = 2
    noise_variance = 10
    Q_noise = np.random.normal(0,noise_variance, Q.shape)
    Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
    Q_noisy = Q+Q_noise #This supposed to be ML output
    Q_lanczos = CG_.project_noisy_ritz_vectors_on_lanczos_space(b, Q_noisy)
    lambda_Q_lanczos = CG_.create_ritz_values(Q_lanczos)
    return CG_.mult_precond_method1(x,Q_lanczos,lambda_Q_lanczos)

x, res_arr= CG.restarted_pcg_manual(b, mult_precond_method_noisy_ritz_vecs_projected_onto_lanczos,max_outer_it, pcg_inner_it, tol, verbose)
#%%
plt.plot(np.log10(res_arr))



#%% Jacobi Iteration
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





#%% PCG tests with RHS from FluidSolver (Incompressible Flow)
# To Ayano

pres_lap = pl.pressure_laplacian(64)
CG = cg.ConjugateGradient(pres_lap.A)
n = 10
b = hf.get_frame(n)   

max_it=300
tol=1.0e-13
verbose=True
num_vectors = 16

Q = CG.create_ritz_vectors(b,num_vectors)
ritz_vals = CG.create_ritz_values(Q)
mult_precond1 = lambda x: CG.mult_precond_method1(x,Q,ritz_vals)
mult_low_rank_precond = lambda x: CG.mult_precond_2_helper(x,Q,ritz_vals)
def mult_diag_precond(x):
    y = x.copy()
    for i in range(len(x)):
        if CG.A[i,i]>0:
            y[i] = y[i]/CG.A[i,i]
    return y

mult_precond_dummy = lambda x: x         
mult_precond_2 = lambda x: CG.mult_precond_2(x,mult_precond1, mult_low_rank_precond)
x_init = mult_low_rank_precond(b)
x, res_arr = CG.pcg_normal(x_init,b, mult_precond1, max_it,tol,verbose)
#%%
res_arr_mult_precond_orig = res_arr.copy()
#%%
#%matplotlib qt
plot_num = 100
plt.plot(np.log10(res_arr_mult_precond2_diag[0:plot_num]) ,label='res_arr_mult_precond2_diag')
plt.plot(np.log10(res_arr_mult_precond2_dummy[0:plot_num]) ,label='res_arr_mult_precond2_dummy')
plt.plot(np.log10(res_arr_mult_precond_orig[0:plot_num]) ,label='res_arr_mult_precond_orig')

plt.legend()

#plt.plot(res_arr)
















