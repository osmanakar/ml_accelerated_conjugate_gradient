project_folder_general = "C:/Users/osman/Desktop/research_teran/python/MLV1/"

import sys
sys.path.insert(1, project_folder_general+'remote/lib/')
import helper_functions as hf
import conjugate_gradient as cg
import matplotlib.pyplot as plt

#%% Noise Added Restarted

#b_orig = f_1D.copy()
#pres_lap = pl.pressure_laplacian(dim-1)
pres_lap = pl.pressure_laplacian(3)
x_orig0 = np.random.rand(pres_lap.m)
b_orig = np.matmul(pres_lap.A, x_orig0)
b_orig = b_orig/np.linalg.norm(b_orig)
b = b_orig.copy()
b_pr = b_orig.copy()
res_arr = []
x_sol = np.zeros(b.shape)

CG =cg.ConjugateGradient(pres_lap.A)
x_init = np.zeros(x_orig0.shape)
max_it = 1
tol = 1.0e-15
verbose=False
sorting = True
num_modes = 10

res_arr = []
for i in range(1):
    print("restarting: i = "+ str(i))
    Q = CG.create_approximate_eigenmodes(b_pr, num_modes, sorting)
    noise_strength = 1;
    noise_variance = 10
    Q_noise = np.random.normal(0,noise_variance, Q.shape)
    Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
    Q_noisy = Q+Q_noise
    #print("lambda_vals_Q = "+ str(lambda_vals_Q))
    lambda_vals = CG.create_lambda_vals(Q)
    mult_precond = lambda x: CG.mult_precond_approximate_eigenmodes(x,Q,lambda_vals)
    x_sol1, res_arr1 = CG.pcg_normal(x_init, b, mult_precond , max_it,tol,verbose)
    x_sol = x_sol + x_sol1
    b = b_orig - np.matmul(pres_lap.A,x_sol)
    b_norm = np.linalg.norm(b)
    b_pr = b/b_norm
    print("residual = "+ str(b_norm))
    res_arr = res_arr + res_arr1[0:max_it]

#aq = hf.AQ(pres_lap.A,mult_precond)
#print(np.linalg.norm(aq-np.identity(aq.shape[0])))
#%%
reduced_idx = pres_lap.reduced_idx()
pres_lap.create_A_reduced()
Q_reduced = Q[:,reduced_idx]
a_red = pres_lap.A_reduced 
mult_precond_red = lambda x: CG.mult_precond_approximate_eigenmodes(x,Q_reduced,lambda_vals)
aq_red = hf.AQ(pres_lap.A_reduced,mult_precond_red)
print(np.linalg.norm(aq_red-np.identity(aq_red.shape[0])))



#%%
dim_ = 10
pres_lap = pl.pressure_laplacian(dim_)
pres_lap.create_A_reduced()
eigvals_A, Q_actual = np.linalg.eig(pres_lap.A_reduced)
idx_ = eigvals_A.argsort()[::-1]   
eigvals_A = eigvals_A[idx_]
Q_actual = Q_actual[:,idx_].transpose()

#%%
x_orig0 = np.random.rand(pres_lap.m - 4)
b_orig0 = np.matmul(pres_lap.A_reduced, x_orig0)
b_orig0 = b_orig0/np.linalg.norm(b_orig0)
b = b_orig0.copy()
b_pr = b_orig0.copy()
res_arr = []
x_sol = np.zeros(b.shape)

CG = cg.ConjugateGradient(pres_lap.A_reduced)
#Q = CG.create_approximate_eigenmodes(b_pr, num_modes, sorting)
x_init = np.zeros(x_orig0.shape)
max_it = 100
tol = 1.0e-15
verbose=False
sorting = True
num_modes = 16

Q_actual_small = Q_actual[0:20]
lambda_vals = CG.create_lambda_vals(Q_actual_small)
mult_precond = lambda x: CG.mult_precond_approximate_eigenmodes(x,Q_actual_small,lambda_vals)
x_sol1, res_arr_Q_actual_full = CG.pcg_normal(x_init, b, mult_precond , max_it,tol,verbose)
b = b_orig0 - np.matmul(pres_lap.A_reduced,x_sol1)
b_norm = np.linalg.norm(b)
print("residual = "+ str(b_norm))

#%% Restarted PCG with actual eigenmodes Test
pres_lap = pl.pressure_laplacian(64)
pres_lap.create_A_reduced()
idx_A_reduced = pres_lap.reduced_idx()
CG = cg.ConjugateGradient(pres_lap.A_reduced)

#we have two choices of RHS:
x_orig1 = np.random.rand(CG.n)
x_orig1 = x_orig1/np.linalg.norm(np.matmul(pres_lap.A_reduced, x_orig1))
b_rand = np.matmul(pres_lap.A_reduced, x_orig1)
#b1 = b1/np.linalg.norm(b1)
b_f10 = f_1D[idx_A_reduced].copy()

b_orig = b_f10

#%%
eigvals_A, Q_actual = np.linalg.eig(pres_lap.A_reduced)
idx_ = eigvals_A.argsort()[::-1]   
eigvals_A = eigvals_A[idx_]
Q_actual = Q_actual[:,idx_].transpose()
#%%
num_modes = 16
max_outer_it = 100
pcg_inner_it = 1
tol = 1.0e-13,
verbose = True

def mult_precond_method_ritz_vecs(CG_, x, b):
    Q = CG_.create_approximate_eigenmodes(b,num_modes)
    lambda_ =  CG_.create_lambda_vals(Q)
    return CG_.mult_precond_approximate_eigenmodes(x,Q,lambda_)

def mult_precond_method_noisy_ritz_vecs(CG_, x, b):
    Q = CG_.create_approximate_eigenmodes(b,num_modes)
    noise_strength = 1;
    noise_variance = 10
    Q_noise = np.random.normal(0,noise_variance, Q.shape)
    #print("noise = "+str(np.linalg.norm(Q_noise)))
    Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
    Q_noisy = Q+Q_noise
    lambda_ =  CG_.create_lambda_vals(Q_noisy)
    return CG_.mult_precond_approximate_eigenmodes(x,Q_noisy,lambda_)

def mult_precond_method_noisy_ritz_vecs_projected_onto_lanczos(CG_, x, b):
    Q = CG_.create_approximate_eigenmodes(b,num_modes)
    noise_strength = 10
    noise_variance = 10
    Q_noise = np.random.normal(0,noise_variance, Q.shape)
    #print("noise = "+str(np.linalg.norm(Q_noise)))
    Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
    Q_noisy = Q+Q_noise
    lanczos_vectors, diagonal0, sub_diagonal0 = CG_.lanczos_iteration(b, num_modes, 1.0e-10)
    projection_coef = np.matmul(Q_noisy,lanczos_vectors.transpose()).transpose()
    Q_noisy_projected_on_lanczos = np.matmul(lanczos_vectors.transpose(),projection_coef).transpose()
    for i in range(num_modes):
        Q_noisy_projected_on_lanczos[i]=Q_noisy_projected_on_lanczos[i]/np.linalg.norm(Q_noisy_projected_on_lanczos[i])
    lambda_Q_noisy_projected_on_lanczos =  CG_.create_lambda_vals(Q_noisy_projected_on_lanczos)
    return CG_.mult_precond_approximate_eigenmodes(x,Q_noisy_projected_on_lanczos,lambda_Q_noisy_projected_on_lanczos)

def mult_precond_method_noisy_ritz_vecs_refined(CG_, x, b):
    Q = CG_.create_approximate_eigenmodes(b,num_modes)
    noise_strength = 1;
    noise_variance = 10
    Q_noise = np.random.normal(0,noise_variance, Q.shape)
    #print("noise = "+str(np.linalg.norm(Q_noise)))
    Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
    Q_noisy = Q+Q_noise
    Q_new = CG_.refine_matrix(Q_noisy)
    lambda_new = CG_.create_lambda_vals(Q_new)
    return CG_.mult_precond_approximate_eigenmodes(x,Q_new,lambda_new)


def mult_precond_method_lanczos_generated_noisy_ritz_vecs_refined(CG_, x, b):
    lanczos_vectors, diagonal0, sub_diagonal0 = CG_.lanczos_iteration(b, num_modes, 1.0e-10)
    Q = CG_.create_approximate_eigenmodes(b,num_modes)
    noise_strength = 1
    noise_variance = 10
    Q_noise_cpt = np.random.normal(0,noise_variance, [num_modes, num_modes])
    Q_noise = np.matmul(lanczos_vectors.transpose(),Q_noise_cpt).transpose()
    Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
    Q_noisy = Q+Q_noise
    [Q, R]=np.linalg.qr(Q_noisy.transpose(), mode="reduced")
    Q = Q.transpose()
    A_tilde = np.matmul(Q,np.matmul(CG_.A, Q.transpose()))
    eigvals,Q0 = np.linalg.eig(A_tilde)
    Q_new = np.matmul(Q.transpose(),Q0).transpose()
    lambda_new = CG_.create_lambda_vals(Q_new)
    return CG_.mult_precond_approximate_eigenmodes(x,Q_new,lambda_new)

#idxs = list(range(0,8))+list(range(CG.n-9,CG.n-1))
idxs = list(range(0,16*250,250))
def mult_precond_method_actual_eigvecs(CG_, x, b):
    Q = Q_actual[idxs]
    lambda_ =  CG_.create_lambda_vals(Q)
    return CG_.mult_precond_approximate_eigenmodes(x,Q,lambda_)

#idxs = list(range(0,8))+list(range(CG.n-9,CG.n-1))
"""
idxs = list(range(0,16*250,250))
def mult_precond_method_new(CG_, x, b):
    ...
    
    return CG_.mult_precond_approximate_eigenmodes(x,Q,lambda_)
"""

#%% Ritz Vectors
x_ritz_vecs, res_arr_ritz_vecs= CG.restarted_pcg_manual(b_orig, mult_precond_method_ritz_vecs,max_outer_it, pcg_inner_it, tol, verbose)
print("mult_precond_method_ritz_vecs")
#%%
res_arr_ritz_vecs_f10 = res_arr_ritz_vecs.copy()
#%% Noisy Ritz Vectors
x_ritz_vecs, res_arr_ritz_vecs= CG.restarted_pcg_manual(b_orig, mult_precond_method_noisy_ritz_vecs,max_outer_it, pcg_inner_it, tol, verbose)
#%%
res_arr_noisy_ritz_vecs_f10 = res_arr_ritz_vecs.copy()
#%% ((Lanczos)Noisy Vectors - refined)
x_ritz_vecs, res_arr_ritz_vecs= CG.restarted_pcg_manual(b_orig, mult_precond_method_lanczos_generated_noisy_ritz_vecs,max_outer_it, pcg_inner_it, tol, verbose)
#%%
res_arr_lanczos_generated_noisy_ritz_vecs_f10 = res_arr_ritz_vecs.copy()

#%% (Noisy Vectors - refined)
x_ritz_vecs, res_arr_ritz_vecs= CG.restarted_pcg_manual(b_orig, mult_precond_method_noisy_ritz_vecs_refined,max_outer_it, pcg_inner_it, tol, verbose)
print("Noisy Vectors - refined")
#%%
res_arr_noisy_ritz_vecs_refined_rand = res_arr_ritz_vecs.copy()

#%%
x_actual_eigmodes, res_arr_actual_eigvecs = CG.restarted_pcg_manual(b_orig, mult_precond_method_noisy_ritz_vecs_projected_onto_lanczos,max_outer_it, pcg_inner_it, tol, verbose)
#%%
res_arr_noisy_ritz_vecs_projected_onto_lanczos_f10 = res_arr_actual_eigvecs.copy()
#plt.plot(np.log10(res_arr2))

#%%



#%%
%matplotlib qt
plot_num = 100
plt.title("Restarted PCG Experiments")
plt.plot(np.log10(res_arr_ritz_vecs_rand[0:plot_num]), label='rand, Ritz Vectors')
plt.plot(np.log10(res_arr_noisy_ritz_vecs_rand[0:plot_num]), label='rand, noisy(%25) ritz vectors')
plt.plot(np.log10(res_arr_noisy_ritz_vecs_refined_rand[0:plot_num]), label='rand, noisy(%25) Ritz Vectors-refined, restarted')
plt.plot(np.log10(res_arr_lanczos_generated_noisy_ritz_vecs_rand[0:plot_num]), label='rand, lanczos-noisy(%25) Ritz Vectors-refined, restarted')


plt.plot(np.log10(res_arr_ritz_vecs_f10[0:plot_num]), label='f10, Ritz Vectors')
plt.plot(np.log10(res_arr_noisy_ritz_vecs_f10[0:plot_num]), label='f10, noisy(%25) Ritz Vectors, restarted')
plt.plot(np.log10(res_arr_noisy_ritz_vecs_refined_f10[0:plot_num]), label='f10, noisy(%25) Ritz Vectors-refined, restarted')
plt.plot(np.log10(res_arr_lanczos_generated_noisy_ritz_vecs_f10[0:plot_num]), label='f10, lanczos-noisy(%25) Ritz Vectors-refined, restarted')
plt.plot(np.log10(res_arr_noisy_ritz_vecs_projected_onto_lanczos_f10[0:plot_num]), label='f10, noisy(%25) Ritz Vectors-projected onto lanczos, restarted')

#plt.plot(np.log10(res_arr_ritz_vecs[0:plot_num]), label='f10, nonrestarted, ritz')


#plt.plot(np.log10(res_arr_lanczos_generated_noisy_ritz_vecs_rand[0:plot_num]), label='rand, lanczos-noisy(%25) Ritz Vectors-refined, restarted')


#plt.ylim([-14,1])
# draw vertical line from (70,100) to (70, 250)
"""
for kk in range(0,int(plot_num/10)+1):
    plt.plot([10*kk, 10*kk], [0, -13], 'k-', lw=0.5)
"""
plt.legend()

# draw diagonal line from (70, 90) to (90, 200)
#plt.plot([20, 0], [20, 200], 'k-')


























