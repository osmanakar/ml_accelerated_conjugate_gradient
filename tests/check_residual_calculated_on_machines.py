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
dim = 64
dim2 = dim**3
#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
#CG128 = cg.ConjugateGradient(pres_lap128.A)
 
#% %
name_sparse_matrix = project_folder_general+"data/A_Sparse_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
#pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = sparse.load_npz(name_sparse_matrix)

#% %
CG = cg.ConjugateGradientSparse(A_sparse)

#%%
hyde01_folder_name = "oak@hyde01.dabh.io:~/projects/ML_preconditioner_project/MLCG_3D_N64/data/test_picking_best_model/"
#project_folder_subname = "MLV32_3_T1_3D_N64_ritz_vectors_20000_10000_V2_for_3D" #1,160
project_folder_subname = "MLV33_1_T1_3D_N64_ritz_vectors_20000_10000_V2_for_3D" #1,45
#project_folder_subname = "MLV33_2_T1_3D_N64_ritz_vectors_20000_10000_V2_for_3D" #1,129
#project_folder_subname = "MLV37_1_5conv_T1_3D_N64_ritz_vectors_20000_10000_V2_for_3D" #1,148; 140, 257
#project_folder_subname = "MLV32_1_T3_3D_N64_A_cos_vectors_faulty_DCT1_20000_10000_V2_for_3D" #1,44
#project_folder_subname = "MLV33_1_T2_3D_N64_A_cos_vectors_20000_10000_V2_for_3D" #1,157; 157,227



epoch_num_start = 1
epoch_num_finish = 45
res_arr_name = "res_arr_matrix_"+project_folder_subname+str(epoch_num_start)+"_"+str(epoch_num_finish)+".npy"
scp_hyde01_file = hyde01_folder_name+res_arr_name
mac_saving_place = "/Users/osmanakar/Desktop/research_related_icloud_temp/3D_residuals_pick_best_model/"
os.system("scp -r "+ scp_hyde01_file + " "+ mac_saving_place)
#% %
with open(mac_saving_place+res_arr_name, 'rb') as f:
    res_arr_matrix_load = np.load(f)
#%%
res_arr_matrix = np.zeros([1000,res_arr_matrix_load.shape[1]])
#%%
res_arr_matrix[epoch_num_start:epoch_num_finish,:]=res_arr_matrix_load[epoch_num_start:epoch_num_finish,:]
#%%
tol = 1.0e-6
idx= 100
good_epochs = []
for i in range(1,epoch_num_finish):
    res0 = res_arr_matrix[i,idx]
    if res0<tol:
        print(i,res0)
        good_epochs = good_epochs + [i]
#%%
%matplotlib qt
plot_num =101
good_epochs = [50,100,120,125,151]

for i in range(len(good_epochs)):
    plt.plot(np.log10(res_arr_matrix[good_epochs[i],0:plot_num]),label="cos-trained-E"+str(good_epochs[i]))

good_epochs = [43]
for i in range(len(good_epochs)):
    plt.plot(np.log10(res_arr_matrix_load[good_epochs[i],0:plot_num]),label="ritz-trained-E"+str(good_epochs[i]))

#plt.plot(np.log10(res_arr_matrix[8,0:plot_num]) ,label='ML+CG-2')
#plt.plot(np.log10(hf.extend_array(res_arr_ML_cg4[0:50], 4)) ,label='ML+CG-4')
#plt.plot(np.log10(hf.extend_array(res_arr_ML_cg10[0:20], 10)) ,label='ML+CG-10')

plt.legend()


#%% 
print("Checking Training & Validation Loss")
#/home/oak/projects/ML_preconditioner_project/MLCG_3D_N64/MLV33_1_T2_3D_N64_A_cos_vectors_20000_10000_V2_for_3D

project_folder_subname = "MLV33_1_T2_3D_N64_A_cos_vectors_20000_10000_V2_for_3D" #1,157
hyde01_folder_name_general = "oak@hyde01.dabh.io:~/projects/ML_preconditioner_project/MLCG_3D_N64/"
scp_hyde01_training_loss = hyde01_folder_name_general+project_folder_subname+"/MLCG_3D_N64_training_loss.npy"
scp_hyde01_validation_loss = hyde01_folder_name_general+project_folder_subname+"/MLCG_3D_N64_validation_loss.npy"
mac_saving_place = "/Users/osmanakar/Desktop/research_related_icloud_temp/training_validation_losses/"
os.system("scp -r "+ scp_hyde01_training_loss + " "+ mac_saving_place)
os.system("scp -r "+ scp_hyde01_validation_loss + " "+ mac_saving_place)

with open(mac_saving_place+"/MLCG_3D_N64_training_loss.npy", 'rb') as f:
    training_loss = np.load(f)

with open(mac_saving_place+"/MLCG_3D_N64_validation_loss.npy", 'rb') as f:
    validation_loss = np.load(f)

#%%
plt.plot((training_loss[50:180]),label="training_loss")
plt.plot((validation_loss[50:180]),label="validation_loss")










#%%
dim = 16
dim2 = dim**3
epoch_num = 50 #460(inc flow) #550 #500
# Best One: 1780
#project_folder_subname = "PFI_MLV9_eigvector_random_N63_50000_dataset"
#project_folder_subname = "MLV6_T1_N511_ritz_vectors_10000_2500"
project_folder_subname = "MLV6_T1_3D_N16_eigen_vectors_10000"

model_3D_N16 = hf.load_model_from_machine(epoch_num, project_folder_general, project_folder_subname,dim, True)

#%%
def model_predict(r):
    r_tf = tf.convert_to_tensor(r.reshape([1,dim,dim]),dtype=tf.float32)
    #return model_N255.predict(r_tf)[0,:,:].reshape([dim2]) #first_residual
    #return model_N255(r_tf,training=False).numpy()[0,:,:].reshape([dim2]) #first_residual
    return model_N255(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:,:].reshape([dim2]) #first_residual

#%%
data_folder_name = "/Users/osmanakar/OneDrive/research_teran/python/data/incompressible_flow_outputs3/output3D_"+str(dim-1)+"/"
n=50
b = hf.get_frame_from_source(n, data_folder_name)
#x_sol, res_arr_ml_generated_cg = CG512.restarted_ML_prediction(b, np.zeros(b.shape), model_N511,max_it,1e-14, True)
#% %
max_it=100
t0=time.time()
x_sol, res_arr_ml_generated_cg_3D = CG3D_16.cg_on_ML_generated_subspace_3D(b, np.zeros(b.shape), model_3D_N16, max_it,1e-14, True)
time_cg_ml = time.time() - t0
print(time_cg_ml)

#%%
print("CG test is running")
t0=time.time()
x_sol_cg, res_arr_cg = CG512.cg_normal(np.zeros(b.shape),b,1000,1e-4,True)
time_cg = time.time() - t0
print("CG took ",time_cg, " secs")

#%% This is when we A-orthonormalize
t0 = time.time()
x_sol, res_arr_ml_generated_cg = CG.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_N255,max_it,tol, True)
time_ml_full = time.time() - t0
print(time_ml_full)

#%%
x_sol, res_arr_ml_iterative = hf.ML_generated_iterative_subspace_solver(b, model_N255, CG,max_it,True)

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
x_sol, res_arr = CG128_block1.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_PFI_V1,max_it,1e-8, True)

#%%
project_folder_subname = "MLV6_T1_N255_ritz_vectors_10000_20000"
n=300
b = hf.get_frame(n,dim,'mac')
max_it=100
res_arr_matrix_256 = np.zeros([1000,max_it+1])
#%%
for i in range(10,140):
    epoch_num = i
    print(epoch_num)
    model_N255 = hf.load_model_from_machine(epoch_num, project_folder_general, project_folder_subname, dim)
    #x_sol, res_arr = hf.ML_generated_iterative_subspace_solver(b, model_PFI_V1, CG, max_it,True)
    x_sol, res_arr = CG.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_N255,max_it,1e-15, True)
    if (res_arr[90]<1e-12):
        print(i)
    print(i)
    res_arr_matrix_256[i]=np.array(res_arr)

#%%
for i in range(1,300):
    epoch_num = i
    print(epoch_num)
    model_PFI_V1 = hf.load_model_from_machine(epoch_num, project_folder_general, project_folder_subname, dim)
    x_sol, res_arr = CG128.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_PFI_V1,max_it,3e-14, True)
    if (res_arr[90]<1e-4):
        print(i) 
    print(i)
    res_arr_matrix_128_V2[i]=np.array(res_arr[0:max_it])


#%%
good_epochs = []
for i in range(1,140):
    if res_arr_matrix_256[i,99]<1e-10:
        print(i,res_arr_matrix_256[i,99])
        good_epochs = good_epochs + [i]




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




