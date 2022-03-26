#%%
import numpy as np
project_folder_general = "C:/Users/osman/Desktop/research_teran/python/MLV1/"
import sys
sys.path.insert(1, project_folder_general+'remote/lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl


#%% ConjugateGradient Test
import conjugate_gradient as cg

A = np.array([[1,2,3,0],
              [2,3,4,10],
              [3,4,5,2],
              [0,10,2,8]])
CG =cg.ConjugateGradient(A)
x = np.array([1,1,2,10])
b = np.matmul(A,x)
xx = np.zeros(4)
max_it = 100
tol = 1.0e-12
verbose=True
xx = CG.cg_normal(xx,b,max_it,tol,verbose)

#%% Lanczos Iteration Test

Q, diag, sub_diag = CG.lanczos_iteration(b,tol,2)
print(np.matmul(Q,Q.transpose()))

#%% Appoximate Eigenvector Creation test:
import pressure_laplacian as pl

pres_lap0 = pl.pressure_laplacian(64)

x_orig0 = np.random.rand(pres_lap0.m)
b0 = np.matmul(pres_lap0.A,x_orig0)
b0 = b0/np.linalg.norm(b0)
CG0 = cg.ConjugateGradient(pres_lap0.A)
num_modes = 50
Q0 = CG0.create_approximate_eigenmodes(b0,num_modes)

x0, res_arr0 = CG0.pcg_normal(np.zeros(pres_lap0.m),b0,Q0)

plt.plot(np.log10(res_arr0))

#%% Restarted PCG Test
import conjugate_gradient as cg
import pressure_laplacian as pl
pres_lap0 = pl.pressure_laplacian(64)

x_orig0 = np.random.rand(pres_lap0.m)
#b0 = np.matmul(pres_lap0.A,x_orig0)
b0 = b0/np.linalg.norm(b0)
b0 = f_1D.copy()
CG0 = cg.ConjugateGradient(pres_lap0.A)
num_modes = 16
max_outer_it = 100
pcg_inner_it = 1
tol = 1.0e-13,
method = "approximate_eigenmodes"
num_modes = 16
verbose = True
x0, res_arr0 = CG0.restarted_pcg(b0, max_outer_it, pcg_inner_it, tol, method , num_modes , verbose)
#%%
%matplotlib qt
plt.plot(np.log10(res_arr0))


#%% create_approximate_eigenmodes_with_ray_quo_iter
print(model_name0)

pres_lap1 = pl.pressure_laplacian(64)
A1 = pres_lap1.A.copy()
b1 = f_1D.copy()
CG1 = cg.ConjugateGradient(A1)
num_modes = 16
lambda_pred = CG1.create_lambda_vals(Q_ML_predicted)
print("lambda_pred1 = " + str(lambda_pred1))
lambda_gt1 =  CG1.create_lambda_vals(Q_gt)
print("lambda_gt1 = " + str(lambda_gt1))
Q_approx_eigvecs1 = CG1.create_approximate_eigenmodes(b1,num_modes,sorting=True)
lambda_approx_eigvecs1 =  CG1.create_lambda_vals(Q_approx_eigvecs1)
print("lambda_gt1 = " + str(lambda_approx_eigvecs1))
#%%
Q_approx_eigvecs_with_ray_quo_iter, lambda_approx_eigvecs_with_ray_quo_iter = CG1.create_approximate_eigenmodes_with_ray_quo_iter(b0,num_modes,lambda_pred, Q_ML_predicted)
#%%
x1, res_arr1 = CG1.pcg_normal(np.zeros(pres_lap1.m),b1,Q_approx_eigvecs_with_ray_quo_iter,max_it=1000)
#%%
plt.plot(np.log10(res_arr1))

#%% AppEig + Diagonal Preconditioner test:
import pressure_laplacian as pl
import conjugate_gradient as cg

pres_lap0 = pl.pressure_laplacian(64)

x_orig0 = np.random.rand(pres_lap0.m)
x_init = np.zeros(x_orig0.shape)
b0 = np.matmul(pres_lap0.A,x_orig0)
b0 = b0/np.linalg.norm(b0)
CG0 = cg.ConjugateGradient(pres_lap0.A)
epsilon = 0.1
Q_approx_eigvecs = CG0.create_approximate_eigenmodes(b0,num_modes)

x0, res_arr0 = CG0.pcg_normalV2(np.zeros(pres_lap0.m),b0,Q_approx_eigvecs, epsilon)
#%%
res_arr_app_eig_and_diag_precond01 = res_arr0.copy() 


#%% Diagonal Preconditioner test:
import pressure_laplacian as pl
import conjugate_gradient as cg

pres_lap0 = pl.pressure_laplacian(64)

x_orig0 = np.random.rand(pres_lap0.m)
x_init = np.zeros(x_orig0.shape)
b0 = np.matmul(pres_lap0.A,x_orig0)
b0 = b0/np.linalg.norm(b0)
CG0 = cg.ConjugateGradient(pres_lap0.A)
epsilon = 1
x0, res_arr0 = CG0.pcg_normal_with_diagonal_preconditioner(x_init,b0,epsilon)

#%%
res_arr_diag_precond1 = res_arr0.copy() 
#plt.plot(np.log10(res_arr0))


#%%
%matplotlib qt
plot_num = 100
#plt.plot(np.log10(res_arr_Q_ML_predicted[0:plot_num]), label='ML predicted')
plt.plot(np.log10(res_arr_Q_gt[0:plot_num]), label='with 16 eigenmodes, nonrestarted')
plt.plot(np.log10(res_arr_cg[0:plot_num]), label='cg')
plt.plot(np.log10(res_arr_restarted[0:plot_num]), label='Restarted With Approximate Eigmodes')
plt.plot(np.log10(res_arr_diag_precond01[0:plot_num]), label='Diag Precond = .1')
plt.plot(np.log10(res_arr_diag_precond02[0:plot_num]), label='Diag Precond = .2')
plt.plot(np.log10(res_arr_diag_precond1[0:plot_num]), label='Diag Precond = 1')
plt.plot(np.log10(res_arr_diag_precond10[0:plot_num]), label='Diag Precond = 10')

plt.plot(np.log10(res_arr_app_eig_and_diag_precond01[0:plot_num]), label='ApproxEig+Diag Precond = .01')
#plt.plot(np.log10(res_arr_noise_strength1[0:plot_num]), label='noise_strength = .1')
#plt.plot(np.log10(res_arr_noise_strength2[0:plot_num]), label='noise_strength = .2')
#plt.plot(np.log10(res_arr_noise_strength3[0:plot_num]), label='noise_strength = .3')
#plt.plot(np.log10(res_arr_noise_strength4[0:plot_num]), label='noise_strength = .4')
#plt.plot(np.log10(res_arr_noise_strength5[0:plot_num]), label='noise_strength = .5')
#plt.plot(np.log10(res_arr_noise_strength10[0:plot_num]), label='noise_strength = 1.0')
#plt.plot(np.log10(res_arr_orthonormalized_noise_strength5[0:plot_num]), label='orhonormalized, noise_strength = 0.5')
#plt.plot(np.log10(res_arr_simul_it_noise_strength5[0:plot_num]), label='simul_it, noise_strength = 0.5')
#plt.plot(np.log10(res_arr_simul_it_noise_strength10[0:plot_num]), label='simul_it, noise_strength = 1')
#plt.plot(np.log10(res_arr_orthonormalized_noise_strength10[0:plot_num]), label='orhonormalized, noise_strength = 1.0')
#plt.plot(np.log10(res_arr_with_qr_noise_strength10[0:plot_num]), label='with QR, noise_strength = 1.0')
#plt.plot(np.log10(res_arr_with_qr_noise_strength5[0:plot_num]), label='with QR, noise_strength = 0.5')
#plt.plot(np.log10(res_arr_with_qr_noise_strength3[0:plot_num]), label='with QR, noise_strength = 0.3')

#plt.plot(np.log10(res_arr[0:plot_num]), label='new')

plt.legend()



























