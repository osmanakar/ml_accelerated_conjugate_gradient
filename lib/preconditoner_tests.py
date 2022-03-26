import numpy as np
lib_path = "C:/Users/osman/OneDrive/research_teran/python/lib/"
#lib_path = "path_to_the_folder_containing_library_functions"

import sys
sys.path.insert(1, lib_path)
import conjugate_gradient as cg
import pressure_laplacian as pl
import matplotlib.pyplot as plt
import helper_functions as hf

dim=16
dim2 = dim**2
#%% Spectral Preconditoner pcg test:
# Spectral Preconditoner chosen from the rhs
num_vectors = 50
pres_lap = pl.pressure_laplacian(64)
CG = cg.ConjugateGradient(pres_lap.A)
x_orig = np.random.rand(pres_lap.m)
#b_orig = np.matmul(pres_lap.A, x_orig)
b_orig = hf.get_frame(10)
b_orig = b_orig/np.linalg.norm(b_orig)

#%%
ritz_vectors = CG.create_ritz_vectors(b_orig,num_vectors)
noise_strength = 0.0;
noise_variance = 10
Q_noise = np.random.normal(0,noise_variance, ritz_vectors.shape)
Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
Q = ritz_vectors+Q_noise

A_tilde = np.matmul(Q,np.matmul(CG.A, Q.transpose()))
x_init = np.matmul(Q.transpose(),np.linalg.solve(A_tilde,np.matmul(Q,b_orig)))
ritz_vals = CG.create_ritz_values(ritz_vectors)
#x_init = CG.mult_precond_2_helper(b_orig,Q,ritz_vals)

print(np.linalg.norm(b_orig-np.matmul(CG.A,x_init)))

#%%
svd_vals = np.linalg.svd(CG.A)

#%%
Q_dense=np.identity(CG.A.shape[0])
for i in range(num_vectors):
    Q_dense = Q_dense + (1/ritz_vals[i]-1)*np.matmul(ritz_vectors[i].reshape([dim2,1]),ritz_vectors[i].reshape([1,dim2]))
#%%
MA = np.matmul(CG.A,Q_dense)
#%%
svd_vals = np.linalg.svd(MA)


#%%
CG = cg.ConjugateGradient(pres_lap.A)
max_it=100
tol_pcg=1.0e-13
verbose=True
x_init=np.zeros(pres_lap.m)
Q = CG.create_ritz_vectors(b,num_vectors)
ritz_vals = CG.create_ritz_values(Q)
#x_init = CG.mult_precond_2_helper(b,Q,ritz_vals)

mult_precond = lambda x: CG.mult_precond_method1(x,Q,ritz_vals)
x, res_arr_pcg1_ = CG.pcg_normal(x_init,b,mult_precond,max_it,tol_pcg,verbose)

#%%
x_orig2 = np.random.rand(pres_lap.m)
b_noise=np.matmul(pres_lap.A, x_orig2)
b_noise=0.001*b_noise/np.linalg.norm(b_noise)
b2 = b+b_noise
b2 = b2/np.linalg.norm(b2)
Q2 = CG.create_ritz_vectors(b2,num_vectors)
ritz_vals2 = CG.create_ritz_values(Q2)
mult_precond2 = lambda x: CG.mult_precond_method1(x,Q2,ritz_vals2)

x_init = CG.mult_precond_2_helper(b,Q2,ritz_vals2)
x2, res_arr_pcg2_ = CG.pcg_normal(x_init,b,mult_precond2,max_it,tol_pcg,verbose)

#%%
%matplotlib qt
plt.plot(np.log10(res_arr_pcg1),label="Q1")
plt.plot(np.log10(res_arr_pcg2),label="Q2")
plt.plot(np.log10(res_arr_pcg2_),label="Q2_")

plt.legend()

#%% Spectral Preconditoner pcg test:
# Spectral Preconditoner chosen from the rhs
pres_lap = pl.pressure_laplacian(64)
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)

#%%
CG = cg.ConjugateGradient(pres_lap.A)
num_vectors = 16
max_it=100
tol_pcg=1.0e-13
verbose=True
x_init=np.zeros(pres_lap.m)
Q = CG.create_ritz_vectors(b,num_vectors)
ritz_vals = CG.create_ritz_values(Q)
#x_init = CG.mult_precond_2_helper(b,Q,ritz_vals)

mult_precond = lambda x: CG.mult_precond_method1(x,Q,ritz_vals)
x, res_arr_pcg1_ = CG.pcg_normal(x_init,b,mult_precond,max_it,tol_pcg,verbose)

#%%
x_orig2 = np.random.rand(pres_lap.m)
b_noise=np.matmul(pres_lap.A, x_orig2)
b_noise=0.001*b_noise/np.linalg.norm(b_noise)
b2 = b+b_noise
b2 = b2/np.linalg.norm(b2)
Q2 = CG.create_ritz_vectors(b2,num_vectors)
ritz_vals2 = CG.create_ritz_values(Q2)
mult_precond2 = lambda x: CG.mult_precond_method1(x,Q2,ritz_vals2)

x_init = CG.mult_precond_2_helper(b,Q2,ritz_vals2)
x2, res_arr_pcg2_ = CG.pcg_normal(x_init,b,mult_precond2,max_it,tol_pcg,verbose)

#%%
%matplotlib qt
plt.plot(np.log10(res_arr_pcg1),label="Q1")
plt.plot(np.log10(res_arr_pcg2),label="Q2")
plt.plot(np.log10(res_arr_pcg2_),label="Q2_")

plt.legend()


#%% PCG tests with RHS from FluidSolver (Incompressible Flow)
# To Ayano

pres_lap = pl.pressure_laplacian(64)
CG = cg.ConjugateGradient(pres_lap.A)
n = 100
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

#%% Stocahastic Gradient Descent Test

pres_lap = pl.pressure_laplacian(dim-1)
CG = cg.ConjugateGradient(pres_lap.A)
dx = 0.01
num_vectors = 8
len_Q = num_vectors*dim2
n = 100
b = hf.get_frame(n,dim)


num_batches = 16
batch_size = int(len_Q/num_batches)
#%%

def first_residual(Q):
    Q = Q.reshape([num_vectors,dim2])
    A_tilde = np.matmul(Q,np.matmul(CG.A, Q.transpose()))
    x_init = np.matmul(Q.transpose(),np.linalg.solve(A_tilde,np.matmul(Q,b)))
    return np.linalg.norm(b - np.matmul(CG.A,x_init))**2

def gradient(Q,idx_list):
    gradient_Q = np.zeros(len_Q)
    fr = first_residual(Q)
    for k in idx_list:
        Q[k] = Q[k]+dx
        gradient_Q[k] = (first_residual(Q) - fr)/dx
        Q[k] = Q[k]-dx
    return gradient_Q


Q_actual = CG.create_ritz_vectors(b,num_vectors)
Q_actual = Q_actual.reshape(dim2*num_vectors)

#%%
Q = np.random.rand(dim2*num_vectors)
#%%
num_epochs =1000
learning_rate = 0.5
for i in range(num_epochs):
    print("Epoch = "+str(i))
    batch_arr = list(range(np.product(Q.shape)))
    np.random.shuffle(batch_arr)
    for j in range(num_batches):
        Q = Q-learning_rate*gradient(Q,batch_arr[batch_size*j:batch_size*(j+1)])
    print(first_residual(Q))



#%% Initial guess ML
x_orig = np.random.rand(pres_lap.m)
b = np.matmul(pres_lap.A, x_orig)
b = b/np.linalg.norm(b)

def gradient_residual(x,idx_list):
    gradient_x = np.zeros((len(x)))
    fr = np.linalg.norm(b-np.matmul(CG.A,x))**2
    for k in idx_list:
        x[k] = x[k]+dx
        gradient_x[k] = (np.linalg.norm(b-np.matmul(CG.A,x))**2 - fr)/dx
        x[k] = x[k]-dx
    return gradient_x 
 
#%%
x = np.random.rand(dim2)
x = x/norm(x)
#%%
num_epochs =1000
learning_rate = 0.00005
batch_size = int(len(x)/num_batches)
for i in range(num_epochs):
    print("Epoch = "+str(i))
    batch_arr = list(range(len(x)))
    np.random.shuffle(batch_arr)
    for j in range(num_batches):
        x = x-learning_rate*gradient_residual(x,batch_arr[batch_size*j:batch_size*(j+1)])
    print(np.linalg.norm(b-np.matmul(CG.A,x)))
    
    

#%%
from joblib import Parallel, delayed
def process(i):
    return i * i
    
results = Parallel(n_jobs=2)(delayed(process)(i) for i in range(10))
print(results)  # prints [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

#%%
import time
from joblib import Parallel, delayed

def countdown(n):
    while n>0:
        n -= 1
    return n


t = time.time()
for _ in range(20):
    print(countdown(10**7), end=" ")
print(time.time() - t)  
# takes ~10.5 seconds on medium sized Macbook Pro


t = time.time()
results = Parallel(n_jobs=2)(delayed(countdown)(10**7) for _ in range(20))
print(results)
print(time.time() - t)
# takes ~6.3 seconds on medium sized Macbook Pro






