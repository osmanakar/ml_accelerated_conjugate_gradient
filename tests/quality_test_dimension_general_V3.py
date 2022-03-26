import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf 
import gc
import time
import scipy.sparse as sparse

project_name = "tests"
project_folder_general = os.path.dirname(os.path.realpath(__file__))+"/"

#%%
sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import matplotlib.pyplot as plt
import helper_functions as hf
            					
#%%
dim = 512
dim2 = dim**2
#%%
pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
A_sparse = pres_lap_sparse.A_sparse
#%%
name_sparse_matrix = project_folder_general+"data/A_Sparse_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)
#%%
CG = cg.ConjugateGradientSparse(A_sparse)
#%%
epoch_num = 62   
#48(MLV6)
#N=64: V6-300, V20-96, V22-130,98 
#N512: Not yet
#project_folder_subname = "PFI_MLV13_eigvector_random_N63"
#project_folder_subname = "MLV19_T4_N255_ritz_vectors_10000_20000"
project_folder_subname = "PFI_MLV22_T1_eigvector_random_N63"
project_folder_subname = "MLV20_T1_N511_ritz_vectors_20000_10000"
ml_model = hf.load_model_from_machine(epoch_num, project_folder_general, project_folder_subname,dim)
print(ml_model.count_params())

#%%
epoch_num = 98   
#48(MLV6)
#N=64: V6-300, V20-96, V22-130,98 
#N=512: Not yet
project_name = "preconditioner_first_iter_MLV1"
project_folder_subname = "PFI_MLV22_T1_eigvector_random_N63"
#project_folder_subname = "MLV20_T1_N511_ritz_vectors_20000_10000"
#ml_model = hf.load_model_from_machine(epoch_num, project_folder_general, project_folder_subname,dim)
ml_model = hf.load_model_from_machine_V2(epoch_num, project_folder_general, project_name, project_folder_subname)
print(ml_model.count_params())
#%% 
epoch_num = 96
#N=64: V6-300, V20-96, V22-130,98 
#N512
project_name = "preconditioner_first_iter_MLV1"
#project_name = "preconditioner_first_iter_MLV1_N511"
project_folder_subname = "PFI_MLV20_T1_eigvector_random_N63" #63
#project_folder_subname = "MLV29_T1_N511_ritz_vectors_20000_10000" #511
model_saving_place = "/Users/osmanakar/Desktop/research_related_icloud_temp/saved_models/"+project_name+"/"  #+project_name+"_json_E"+str(epoch_num)+"/"
#machine="hyde01"
machine="legion"

#48(MLV6)
#N=64: V6-300, V20-96, V22-130,98 
#N=512: Not yet
model =  hf.load_model_from_machine_V3(epoch_num, model_saving_place, project_name, project_folder_subname, machine)
#%%
epoch_num = 50
#N=64: V6-300, V20-96, V22-130,98 
#N512
#project_name = "preconditioner_first_iter_MLV1"
project_name = "preconditioner_first_iter_MLV1_N511"
#project_folder_subname = "PFI_MLV20_T1_eigvector_random_N63" #63
project_folder_subname = "MLV29_T1_N511_ritz_vectors_20000_10000" #511
model_saving_place = "/Users/osmanakar/Desktop/research_related_icloud_temp/saved_models/"+project_name+"/"  #+project_name+"_json_E"+str(epoch_num)+"/"
machine="hyde01"
#machine="legion"

#48(MLV6)
#N=64: V6-300, V20-96, V22-130,98 
#N=512: Not yet
model =  hf.load_model_from_machine_V3(epoch_num, model_saving_place, project_name, project_folder_subname, machine)


#%%
print(ml_model.layers[10].weights[0][:,:,0,0])

#%% Getting RHS: From Frame or Creating one 
data_folder_name = project_folder_general+"../" +project_name+"/data/rhs_from_incompressible_flow/"
b = hf.get_frame_from_source(300, data_folder_name)
#%%
rand_vec_x = np.random.normal(0,1, [dim2])
b = CG.multiply_A_sparse(rand_vec_x)
b = b/np.linalg.norm(b)

#%%
max_it=100
model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual
tol = 1e-12

print("dumming calling for loading")
x_sol, res_arr_ml_generated_cg = CG.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_predict, 3,0.1, True)

t0=time.time()
x_sol, res_arr_ml_generated_cg = CG.cg_on_ML_generated_subspace_test5(b, np.zeros(b.shape), model_predict, max_it,tol, True)
time_cg_ml = round((time.time() - t0)*1000)
print("MLCG took ",time_cg_ml," secs")
#%%
t0=time.time()
x_sol, res_arr_ml_generated_cg = CG.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_predict, max_it,tol, True)
time_cg_ml = round((time.time() - t0)*1000)
print("MLCG took ",time_cg_ml," secs")
#%%
res_arr_ml_generated_cg_MLV22 = res_arr_ml_generated_cg.copy()

#%%
tol = 1e-12
orthonormalization_num=50
true_norm_calculation = True
output_search_directions = False
output_mlcg = CG.cg_on_ML_generated_subspace_A_normal_general(b, np.zeros(b.shape), model_predict, orthonormalization_num, max_it,tol, True,true_norm_calculation, output_search_directions)

#%%
print("CG test is running")
t0=time.time()
x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,1000,tol,True)
time_cg = round((time.time() - t0)*1000)
print("CG took ",time_cg, " ms")


#%% Choosing the best epoch:
project_folder_subname = "PFI_MLV20_T2_eigvector_random_N63"
#%%
max_it=100
res_arr_matrix_512 = np.zeros([1000,max_it+1])
#%%
for i in range(1,100):
    epoch_num = i
    print(epoch_num)
    #ml_model = hf.load_model_from_machine(epoch_num, project_folder_general, project_folder_subname, dim)
    model =  hf.load_model_from_machine_V3(epoch_num, model_saving_place, project_name, project_folder_subname, machine)

    #x_sol, res_arr = hf.ML_generated_iterative_subspace_solver(b, model_PFI_V1, CG, max_it,True)
    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual
    x_sol, res_arr = CG.cg_on_ML_generated_subspace_test5(b, np.zeros(b.shape), model_predict,max_it,1e-14, True)
    res_arr_matrix_512[i,0:len(res_arr)]=np.array(res_arr)

#%%
good_epochs = []
for i in range(1,100):
    idx_ = 20
    tol_ = 1.0e-13
    if res_arr_matrix_64[i,idx_]<tol_:
        print(i,res_arr_matrix_64[i,idx_])
        good_epochs = good_epochs + [i]



#%% Plotting: 
#Uncomment this if you want to see plots
import matplotlib.pyplot as plt
res_fun = lambda res_arr: np.log10(res_arr)
%matplotlib qt
plot_num = 100

#for i in good_epochs:
#    plt.plot(res_fun(res_arr_matrix_64[i,0:plot_num]) ,label=str(i))
#plt.plot(res_fun(res_arr_matrix_64[318,0:plot_num]) ,label='test')
#plt.plot(res_fun(res_arr_matrix_64[96,0:plot_num]) ,label='test')
#plt.plot(res_fun(res_arr_matrix_64[295,0:plot_num]) ,label='test')
#plt.plot(res_fun(res_arr_matrix_64[300,0:plot_num]) ,label='test')
#plt.plot(res_fun(res_arr_matrix_64[318,0:plot_num]) ,label='test')


plt.plot(res_fun(res_arr_ml_generated_cg_MLV6[0:plot_num]) ,label='V6')
plt.plot(res_fun(res_arr_ml_generated_cg_MLV20[0:plot_num]) ,label='V20')
plt.plot(res_fun(res_arr_ml_generated_cg_MLV22[0:plot_num]) ,label='V22-E130, num_param=453505')
#plt.plot(res_fun(res_arr_ml_generated_cg_z[0:plot_num]) ,label='ML with A normalization')
plt.plot(res_fun(res_arr_cg[0:plot_num]),'b',label='cg')
#plt.plot(res_fun(res_arr_pcg16[0:plot_num]) ,label='pcg-16')
#plt.plot(res_fun(res_arr_pcg_jacobi[0:plot_num]) ,label='pcg-jacobi')
#plt.plot(res_fun(res_arr_ldlt[0:plot_num]) ,label='pcg-ldlt')
#plt.plot(res_fun(res_arr_restarted16[0:plot_num]) ,label='pcg-restarted-16')

plt.title("N = "+str(dim)+", log scale")
plt.legend()
#"""





#%% Multiple_inputs































#%%
#Q_actual16 =  CG64.create_ritz_vectors(b,16)
#ritz_vals16 = CG64.create_ritz_values(Q_actual16)
#print("PCG test with spectral preconditoner with 16 ritz values (non-restarted) is running...")
#mult_precond16 = lambda x: CG64.mult_precond_method1(x,Q_actual16,ritz_vals16)
#x, res_arr_pcg16 = CG64.pcg_normal(np.zeros(b.shape),b,mult_precond16,max_it,1e-10,verbose)
#print("PCG test with Jacobi preconditoner is running...")
#x, res_arr_pcg_jacobi = CG.pcg_normal(np.zeros(b.shape),b,CG.mult_diag_precond,max_it,1e-10,verbose)
print("CG test is running")
t0=time.time()
x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,1000,1e-14,True)
time_cg = time.time() - t0
print("CG took ",time_cg, " secs")
#% % ldlt test. This is a bit slow. So intead I have result for n=300 frame precomputed, and it is loading
#_, res_arr_ldlt = CG64.ldlt_pcg(b, max_it, 1.0e-15)
#with open(dir_path + "/saved_residuals/N63/res_arr_ldlt_frame"+str(n)+".npy", 'rb') as f:
#    res_arr_ldlt = np.load(f)

#%%
#with open(dir_path + "/saved_residuals/N63/res_arr_ldlt_frame"+str(n)+".npy", 'wb') as f:
#    np.save(f, res_arr_ldlt)

#%%
print("Restarted PCG with 16 Ritz Vectors is running...")
def mult_precond_method(CG_, x, b):
    Q = CG_.create_ritz_vectors(b,16)
    lambda_ =  CG_.create_ritz_values(Q)
    return CG_.mult_precond_method1(x,Q,lambda_)

x, res_arr_restarted16= CG64.restarted_pcg_manual(b, mult_precond_method, 100, 1, 1e-14, verbose)
#%% Plotting: 
#Uncomment this if you want to see plots
#"""
import matplotlib.pyplot as plt
def res_fun(res_arr):
    return np.log10(res_arr)
    #return res_arr
%matplotlib qt
plot_num = 100
#plt.plot(res_fun(res_arr[0:plot_num]) ,label='test')
#plt.plot(res_fun(res_arr_ml_iterative[0:plot_num]) ,label='ML')
plt.plot(res_fun(res_arr_ml_generated_cg[0:plot_num]) ,label='ML with A normalization')
#plt.plot(res_fun(res_arr_ml_generated_cg_z[0:plot_num]) ,label='ML with A normalization')
plt.plot(res_fun(res_arr_cg[0:plot_num]),'b',label='cg')
#plt.plot(res_fun(res_arr_pcg16[0:plot_num]) ,label='pcg-16')
#plt.plot(res_fun(res_arr_pcg_jacobi[0:plot_num]) ,label='pcg-jacobi')
#plt.plot(res_fun(res_arr_ldlt[0:plot_num]) ,label='pcg-ldlt')
#plt.plot(res_fun(res_arr_restarted16[0:plot_num]) ,label='pcg-restarted-16')
plt.title("N = 256, log scale")
plt.legend()
#"""


#%%

#%%



#%% Testing different methods:
n=400
max_it=100
#x_rand = np.random.normal(0,1, [dim2])
#b = np.matmul(pres_lap.A,x_rand)
b = hf.get_frame(n,128,'mac')
#b = x_test_np[n].reshape(dim2)
#b = b/np.linalg.norm(b)
#b_tf = tf.convert_to_tensor(b.reshape([1,dim,dim,1]),dtype=tf.float32)
#Q_ML = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()
#Q_actual4 = CG.create_ritz_vectors(b,num_vectors)
#Q_actual16 =  CG.create_ritz_vectors(b,16)

#testing Q_ML
b1 = b.copy()
x_sol = np.zeros(b.shape)
res_arr = []

#%%
dim=128
dim2=dim**2
for i in range(100):
    b1_norm = np.linalg.norm(b1)
    b1_normalized = b1/b1_norm
    b_tf = tf.convert_to_tensor(b1_normalized.reshape([1,dim,dim,1]),dtype=tf.float32)
    #Q_ = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()
    q = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2])
    #x, res = first_residual(b1,Q_)
    x = first_residual_fast128_old(b1,q)
    x_sol = x_sol+x
    #distrubition_b1 = np.matmul(eigenvectors_A,b1)
    #plt.bar(eigvals_A_real,distrubition_b1)
    #bar_range = 32
    #distrubition_bar = hf.create_plot_bar_arr(distrubition_b1,bar_range)
    #plt.bar(list(range(bar_range)),distrubition_bar)
    b1 = b- np.matmul(pres_lap128.A,x_sol)
    #b1 = b - tf.sparse.sparse_dense_matmul(A_sparse128, tf.reshape(tf.constant(x_sol,dtype=tf.float32), [dim2,1])).numpy().reshape(dim2)
    res_arr = res_arr + [b1_norm]
    print(i, b1_norm)

#%%
res_arr_N127_new_E290 = res_arr.copy()
#%%
res_arr_N127_E29 = res_arr.copy()
#%%
res_arr_N_127_20000_V5_T4_A_normalized2_E46 = res_arr.copy()
#%%
def res_fun(res_arr):
    return np.log10(res_arr)
    #return res_arr
#%%
%matplotlib qt
plot_num = max_it
plt.plot(res_fun(res_arr[0:plot_num]) ,label='res_arr-test')

#plt.plot(res_fun(res_arr_E2[0:plot_num]) ,label='ML-E2')
#plt.plot(res_fun(res_arr_E3[0:plot_num]) ,label='ML-E3')
#plt.plot(res_fun(res_arr_E4[0:plot_num]) ,label='ML-E4')
#plt.plot(res_fun(res_arr_E5[0:plot_num]) ,label='ML-E5')
#plt.plot(res_fun(res_arr_E6[0:plot_num]) ,label='ML-E6')
plt.plot(res_fun(res_arr_cg[0:plot_num]) ,label='ML-E7')
#plt.plot(res_fun(res_arr_N127_new_E29[0:plot_num]) ,label='ML-new-E29')
#plt.plot(res_fun(res_arr_N127_new_E43[0:plot_num]) ,label='ML-new-E43')
#plt.plot(res_fun(res_arr_matrix_128[46,0:plot_num]) ,label='ML model')
#plt.plot(res_fun(res_arr_N_127_20000_V5_T4_A_normalized1_E46[0:plot_num]) ,label='ML-V5_T4-E46-A_normalized1')
#plt.plot(res_fun(res_arr_N_127_20000_V5_T4_A_normalized2_E46[0:plot_num]) ,label='ML-V5_T4-E46-A_normalized2')
#plt.plot(res_fun(res_arr_N_127_20000_V5_T4_A_normalized2_E46[0:plot_num]) ,label='ML model with A-normalization')
#plt.plot(res_fun(res_arr_N_127_20000_V5_T4_A_normalized3_E46[0:plot_num]) ,label='ML-V5_T4-E46-A_normalized3')

#plt.plot(res_fun(res_arr_matrix_128[69,0:plot_num]) ,label='ML-test')



#plt.plot(res_fun(res_arr_cg_N127[0:plot_num]),'b',label='cg')
#plt.plot(res_fun(res_arr_pcg16_N127[0:plot_num]) ,label='pcg-16')
#plt.plot(res_fun(res_arr_pcg_jacobi_N127[0:plot_num]) ,label='jacobi')
#plt.plot(res_fun(res_arr_ldlt_N127[0:plot_num]),'b',label='ldlt')
plt.title("N = "+str(dim-1))
plt.legend()

#plt.plot(np.log10(res_arr4[0:plot_num]),'k' ,label='ML-4')
#plt.plot(np.log10(res_arr_[0:plot_num]) ,label='initial_guess old')
#plt.plot(np.log10(res_arr_cg[0:plot_num]),'b',label='cg')
#plt.plot(np.log10(res_arr_pcg16[0:plot_num]) ,label='pcg-16')
#plt.plot(np.log10(res_arr_pcg_jacobi[0:plot_num]) ,label='jacobi')
#plt.plot(np.log10(res_arr_ldlt[0:plot_num]) ,label='ldlt')


#%%
#ritz_vals4 = CG.create_ritz_values(Q_actual4)
#mult_precond4 = lambda x: CG.mult_precond_method1(x,Q_actual4,ritz_vals4)
#x, res_arr_pcg4 = CG.pcg_normal(np.zeros(b.shape),b,mult_precond4,max_it,1e-10,True)
CG = cg.ConjugateGradient(A_N128_block1)
#%%
#ritz_vals16 = CG.create_ritz_values(Q_actual16)
#mult_precond16 = lambda x: CG.mult_precond_method1(x,Q_actual16,ritz_vals16)
#x, res_arr_pcg16 = CG.pcg_normal(np.zeros(b.shape),b,mult_precond16,max_it,1e-10,True)
x, res_arr_pcg_jacobi = CG.pcg_normal(np.zeros(b.shape),b,CG.mult_diag_precond,max_it,1e-10,True)
_, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,max_it,1e-14,True)
#%%
_, res_arr_ldlt = CG.ldlt_pcg(b, max_it, 1.0e-15)

#%%
def mult_precond_method(CG_, x, b):
    Q = CG_.create_ritz_vectors(b,16)
    lambda_ =  CG_.create_ritz_values(Q)
    return CG_.mult_precond_method1(x,Q,lambda_)

x, res_arr_restarted16= CG.restarted_pcg_manual(b, mult_precond_method, 100, 1, 1e-14, True)
#plt.plot(np.log10(res_arr))

#%%
n=10
b = x_test_np[n].reshape(dim2)
b = b/np.linalg.norm(b)
b_tf = tf.convert_to_tensor(b.reshape([1,dim,dim,1]),dtype=tf.float32)
Q = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,1]).transpose()
x_init, res = first_residual(b,Q)
Q_actual = CG.create_ritz_vectors(b,16)
#_,res2 = first_residual(b,Q_actual)
_, res_arr_cg_init = CG.cg_normal(x_init,b,100,1e-14,True)
res_arr_cg_init = [1]+res_arr_cg_init
#%%
x, res_arr_pcg16_init = CG.pcg_normal(x_init,b,mult_precond16,100,1e-10,True)
res_arr_pcg16_init = [1]+res_arr_pcg16_init
#%%
%matplotlib qt
plot_num = max_it
plt.plot(np.log10(res_arr1[0:plot_num]),'r' ,label='ML-1')
#plt.plot(np.log10(res_arr4[0:plot_num]),'k' ,label='ML-4')
#plt.plot(np.log10(res_arr_[0:plot_num]) ,label='initial_guess old')
plt.plot(np.log10(res_arr_cg[0:plot_num]),'b',label='cg')
plt.plot(np.log10(res_arr_pcg16[0:plot_num]) ,label='pcg-16')
plt.plot(np.log10(res_arr_pcg_jacobi[0:plot_num]) ,label='jacobi')
plt.plot(np.log10(res_arr_ldlt[0:plot_num]) ,label='ldlt')

#plt.plot(np.log10(res_arr_pcg4[0:plot_num]) ,label='res_arr_pcg 4')
#plt.plot(np.log10(res_arr_restarted4[0:plot_num]) ,label='restarted-4')
#plt.plot(np.log10(res_arr_restarted16[0:plot_num]) ,label='restarted-16')
#plt.plot(np.log10(res_arr_cg_init[0:plot_num]) ,label='ML+CG')
#plt.plot(np.log10(res_arr_ML_cg2[0:plot_num]) ,label='ML+CG-2')
#plt.plot(np.log10(res_arr_ML_cg4[0:plot_num]) ,label='ML+CG-4')
#plt.plot(np.log10(res_arr_ML_cg10[0:plot_num]) ,label='ML+CG-10')


#plt.plot(np.log10(hf.extend_array(res_arr_ML_cg2[0:plot_num], 2)) ,label='ML+CG-2')
#plt.plot(np.log10(hf.extend_array(res_arr_ML_cg4[0:50], 4)) ,label='ML+CG-4')
#plt.plot(np.log10(hf.extend_array(res_arr_ML_cg10[0:20], 10)) ,label='ML+CG-10')

plt.legend()


#%% Testing cg with different lanczos -- 
pres_lap = pl.pressure_laplacian(dim-1)
CG = cg.ConjugateGradient(pres_lap.A)

n=100
x_rand = np.random.normal(0,1, [dim2])
#b = np.matmul(pres_lap.A,x_rand)
#b = hf.get_frame(n,dim)
b = x_test_np[n].reshape(dim2)
b = b/np.linalg.norm(b)
b_tf = tf.convert_to_tensor(b.reshape([1,dim,dim,1]),dtype=tf.float32)
Q_ML = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()

for i in range(1,2):
    x = CG.cg_with_different_lanczos_base_slow(np.zeros(b.shape), b, Q_ML[0],i)
    print(np.linalg.norm(b-np.matmul(CG.A,x)))


#%% 
n=100
x_rand = np.random.normal(0,1, [dim2])
#b = np.matmul(pres_lap.A,x_rand)b = hf.get_frame(n,dim)
b = x_test_np[n].reshape(dim2)
#b = b_rhs[0].copy()
b = b/np.linalg.norm(b)
b_tf = tf.convert_to_tensor(b.reshape([1,dim,dim,1]),dtype=tf.float32)
Q = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()
_, res = first_residual(b,Q)
Q_actual = CG.create_ritz_vectors(b,16)
_,res2 = first_residual(b,Q_actual)
_, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,10,1e-14,True)
print(res,res2,res_arr_cg[1])



#%% Testing different methods:
n=300
max_it=100
x_rand = np.random.normal(0,1, [dim2])
#b = np.matmul(pres_lap.A,x_rand)
b = hf.get_frame(n,dim,'mac')
#b = x_test_np[n].reshape(dim2)
b = b/np.linalg.norm(b)
b_tf = tf.convert_to_tensor(b.reshape([1,dim,dim,1]),dtype=tf.float32)
Q_ML = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()
#Q_actual4 = CG.create_ritz_vectors(b,num_vectors)
Q_actual16 =  CG.create_ritz_vectors(b,16)

#testing Q_ML
b1 = b.copy()
x_sol = np.zeros(b.shape)
res_arr = [1]
#%%
for i in range(1):
    b1_norm = np.linalg.norm(b1)
    b1_normalized = b1/b1_norm
    b_tf = tf.convert_to_tensor(b1_normalized.reshape([1,dim,dim,1]),dtype=tf.float32)
    Q_ = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()
    x1 = CG.cg_with_different_lanczos_base_slow(np.zeros(b.shape), b1, Q_[0],1)
    #x1 = CG.cg_with_different_lanczos_base_slow(np.zeros(b.shape), b1, b1,3)
    x_sol = x_sol+x1
    res = np.linalg.norm(b-np.matmul(CG.A,x_sol))    
    b1 = b-np.matmul(pres_lap.A,x_sol)
    res_arr = res_arr + [res]
    print(res)
    
#%%
b1_norm = np.linalg.norm(b1)
b1_normalized = b1/b1_norm
b_tf = tf.convert_to_tensor(b1_normalized.reshape([1,dim,dim,1]),dtype=tf.float32)
Q_ = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()
for i in range(1,11):
    x1 = CG.cg_with_different_lanczos_base_slow(np.zeros(b.shape), b1, Q_[0],i)
    #x1 = CG.cg_with_different_lanczos_base_slow(np.zeros(b.shape), b1, b1,3)
    res = np.linalg.norm(b-np.matmul(CG.A,x1))    
    print("ML Lanczos base iteration = "+str(i)+' residual = '+str(res))

#%%
res_arr_ML_cg4 = res_arr.copy()


#%% ML output pcg
n=100
x_rand = np.random.normal(0,1, [dim2])
#b = np.matmul(pres_lap.A,x_rand)
b = hf.get_frame(n,dim,'mac')
#b = x_test_np[n].reshape(dim2)
b = b/np.linalg.norm(b)
b_tf = tf.convert_to_tensor(b.reshape([1,dim,dim,1]),dtype=tf.float32)
Q = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()

A_tilde = np.matmul(Q,np.matmul(CG.A, Q.transpose()))
W1 = np.matmul(Q.transpose(),np.linalg.solve(A_tilde,Q))
W2 = np.matmul(Q.transpose(),Q)
mult_precond = lambda x: x + np.matmul(W1,x)

#%%
Q = CG.create_ritz_vectors(b,16)
A_tilde = np.matmul(Q,np.matmul(CG.A, Q.transpose()))
def mult_precond2(x):
    x1 = np.matmul(Q.transpose(),np.linalg.solve(A_tilde,np.matmul(Q,x)))
    x2 = np.matmul(Q.transpose(),np.matmul(Q,x))
    return x + x1 - x2
x, res_arr_pcg_Q = CG.pcg_normal(np.zeros(b.shape),b,mult_precond2,max_it,1e-10,True)
#%%
ritz_vectors = CG.create_ritz_values(Q)
mult_precond1 = lambda x: CG.mult_precond_method1(x,Q,ritz_vectors)
x, res_arr_pcg_Q1 = CG.pcg_normal(np.zeros(b.shape),b,mult_precond1,max_it,1e-10,True)


#%%
max_it = 100
n=40
#x_rand = np.random.normal(0,1, [dim2])
#b = np.matmul(pres_lap.A,x_rand)
b = hf.get_frame(n,dim,'mac')
#b = x_test_np[n].reshape(dim2)
b = b/np.linalg.norm(b)
b1 = b.copy()
x_sol = np.zeros(b.shape)
res_arr = [1]
for i in range(max_it):
    b1_norm = np.linalg.norm(b1)
    b1_normalized = b1/b1_norm
    b_tf = tf.convert_to_tensor(b1_normalized.reshape([1,dim,dim,1]),dtype=tf.float32)
    Q = model_PFI_V1.predict(b_tf)[0,:,:,:].reshape([dim2,num_vectors]).transpose()
    x, res = first_residual(b1,Q)
    x_sol = x_sol+x
    b1 = b-np.matmul(pres_lap.A,x_sol)
    res_arr = res_arr + [res]
    print(res)

#%%
num_vectors_ = 16
Q_actual = CG.create_ritz_vectors(b,num_vectors_)
ritz_vals = CG.create_ritz_values(Q_actual)
mult_precond = lambda x: CG.mult_precond_method1(x,Q_actual,ritz_vals)
x, res_arr_pcg_16 = CG.pcg_normal(np.zeros(b.shape),b,mult_precond,max_it,1e-10,True)
_, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,max_it,1e-14,True)



#%%
dim = 128
res_data_folder = project_folder_general+"../preconditioner_first_iter_MLV1_N127/ldlt_on_machines/"
with open(res_data_folder + "saved_residuals/res_arr_cg_N"+str(dim)+"_frame"+str(n)+".npy", 'rb') as f:
    res_arr_cg_N127 = np.load(f)
with open(res_data_folder + "saved_residuals/res_arr_pcg16_N"+str(dim)+"_frame"+str(n)+".npy", 'rb') as f:
    res_arr_pcg16_N127 = np.load(f)
with open(res_data_folder + "saved_residuals/res_arr_pcg_jacobi_N"+str(dim)+"_frame"+str(n)+".npy", 'rb') as f:
    res_arr_pcg_jacobi_N127 = np.load(f)
with open(res_data_folder + "saved_residuals/res_arr_ldlt_N"+str(dim)+"_frame"+str(n)+".npy", 'rb') as f:
    res_arr_ldlt_N127 = np.load(f)
#with open(project_folder_general + "saved_residuals/res_arr_restarted16_N"+str(dim)+"_frame"+str(n)+".npy", 'rb') as f:
#    np.save(f, res_arr_restarted16)

#%%

#plt.plot(np.log10(res_arr4[0:plot_num]),'k' ,label='ML-4')
plt.plot(np.log10(res_arr1[0:plot_num]) ,label='initial_guess old')
plt.plot(np.log10(res_arr_cg[0:plot_num]),'b',label='cg')
plt.plot(np.log10(res_arr_pcg16[0:plot_num]) ,label='pcg-16')
plt.plot(np.log10(res_arr_pcg_jacobi[0:plot_num]) ,label='jacobi')
plt.plot(np.log10(res_arr_ldlt128[0:plot_num]) ,label='ldlt')
    
plt.legend()



#%% Plot Training and Validation loss
#scp osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1_N127/MLV5_T1_N127/preconditioner_first_iter_MLV1_N127_training_loss.npy ./model_details
#scp osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1_N127/MLV5_T1_N127/preconditioner_first_iter_MLV1_N127_validation_loss.npy  ./model_details

#%matplotlib qt
plot_num = 500
plot_num_lower = 0
os.system("scp osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1_N127/MLV5_T1_N127/preconditioner_first_iter_MLV1_N127_training_loss.npy " +project_folder_general+"model_details")
os.system("scp osman@legion.math.ucla.edu:~/projects/ML_preconditioner_project/preconditioner_first_iter_MLV1_N127/MLV5_T1_N127/preconditioner_first_iter_MLV1_N127_validation_loss.npy "+project_folder_general+"model_details")


with open(project_folder_general +  "model_details/preconditioner_first_iter_MLV1_N127_training_loss.npy", 'rb') as f:
    training_loss = np.load(f)
with open(project_folder_general +  "model_details/preconditioner_first_iter_MLV1_N127_validation_loss.npy", 'rb') as f:
    validation_loss = np.load(f)
plt.title("Machine Learning Losses: ")
#plt.plot(np.log10(MLV13_training_loss[0:plot_num]),'k' ,label='training')
#plt.plot(np.log10(MLV13_validation_loss[0:plot_num]),'r' ,label='validation')
plt.plot(training_loss[plot_num_lower:plot_num],'k' ,label='training')
plt.plot(validation_loss[plot_num_lower:plot_num],'r',label='validation')
plt.legend()


"""
#%% Saving and loading in keras format
from tensorflow.keras.models import Sequential, save_model, load_model
filepath = './saved_model_test'
save_model(model_N255, filepath)
#%%
model = load_model(filepath, compile = True)
#%%
def model_predict(r):
    r_tf = tf.convert_to_tensor(r.reshape([1,dim,dim,1]),dtype=tf.float32)
    return model.predict(r_tf)[0,:,:].reshape([dim2]) #first_residual
"""





