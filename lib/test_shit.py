project_folder_general = "C:/Users/osman/Desktop/research_teran/python/MLV1/"

import sys
sys.path.insert(1, project_folder_general+'remote/lib/')
import helper_functions as hf
import conjugate_gradient as cg
import matplotlib.pyplot as plt
import pressure_laplacian as pl

pres_lap1 = pl.pressure_laplacian(2)
a1 = pres_lap1.A

#%%
pres_lap = pl.pressure_laplacian(64)
CG = cg.ConjugateGradient(pres_lap.A)
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
    Q_noise = np.random.normal(1,noise_variance, Q.shape)
    Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
    Q_noisy = Q+Q_noise #This supposed to be ML output
    Q_lanczos = CG_.project_noisy_ritz_vectors_on_lanczos_space(b, Q_noisy)
    lambda_Q_lanczos = CG_.create_ritz_values(Q_lanczos)
    return CG_.mult_precond_method1(x,Q_lanczos,lambda_Q_lanczos)

x, res_arr= CG.restarted_pcg_manual(b, mult_precond_method_noisy_ritz_vecs_projected_onto_lanczos,max_outer_it, pcg_inner_it, tol, verbose)
#%%
plt.plot(np.log10(res_arr))

#%%

Q = CG.create_ritz_vectors(b,num_vectors)
noise_strength = 2
noise_variance = 10
Q_noise = np.random.normal(1,noise_variance, Q.shape)
Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
Q_noisy = Q+Q_noise #This supposed to be ML output
Q_lanczos = CG.project_noisy_ritz_vectors_on_lanczos_space(b, Q_noisy)
lambda_Q_lanczos = CG.create_ritz_values(Q_lanczos)

#%%
x_test = np.zeros(CG.n)
zero_idx = pres_lap.zero_indexes()
x_test[zero_idx[1]] = 1
test = np.matmul(Q_lanczos,x_test)