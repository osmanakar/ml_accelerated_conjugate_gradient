project_name = "MLCG_3D_N64"
#project_folder_general = "/home/osman/projects/ML_preconditioner_project/"+project_name+"/"
project_folder_general = "/home/oak/projects/ML_preconditioner_project/"+project_name+"/"
project_data_folder = "/home/oak/projects/ML_preconditioner_project/data/"+project_name+"/"


import sys
import os
#project_folder_subname = os.path.basename(os.getcwd())
import numpy as np
import tensorflow as tf
import gc
import scipy.sparse as sparse
#import matplotlib.pyplot as plt
from numpy.linalg import norm
import time

sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf

dim = 64
dim2 = dim**3

#%%
project_folder_subname = sys.argv[1]
tol = np.double(sys.argv[2])
epoch_num = int(sys.argv[3])
b_rhs_n = int(sys.argv[4])
max_it_cg = int(sys.argv[5])

#%%
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
#pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = sparse.load_npz(name_sparse_matrix)

#%%
CG = cg.ConjugateGradientSparse(A_sparse)
#%% Getting RHS for the Testing
if b_rhs_n == 0:
    rand_vec_x = np.random.normal(0,1, [dim2])
    b = CG.multiply_A_sparse(rand_vec_x)
    b = b/np.linalg.norm(b)
else: 
    data_folder_name = project_folder_general+"data/rhs_from_incompressible_flow/output3d_64_from_ayano/"
    n=b_rhs_n
    b = hf.get_frame_from_source(n, data_folder_name)

#%%
model = hf.load_model_from_source(project_folder_general+project_folder_subname+"/saved_models/"+project_name+"_json_E"+str(epoch_num)+"/")
print("model has parameters is ",model.count_params())
model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual

#%%
zero_rows = CG.get_zero_rows()
zero_vectors = np.ones([1,dim2])
zero_vectors[0,zero_rows] = 0.0
zero_vectors[0] = zero_vectors[0]/np.linalg.norm(zero_vectors[0])
#%%
max_it=100
print("Dummy calling")
x_sol, res_arr_ml_generated_cg_3D = CG.cg_on_ML_generated_subspace(b, np.zeros(b.shape), model_predict, max_it, 0.01, False)

t0=time.time()
x_sol, res_arr_ml_generated_cg_3D = CG.cg_on_ML_generated_subspace_test5(b, np.zeros(b.shape), model_predict, max_it,tol, True)
time_cg_ml = time.time() - t0
print("ML took ", time_cg_ml," secs.")


#%%
print("CG test is running - with residual projection")
t0=time.time()
x_sol_cg, res_arr_cg = CG.cg_normal_test2(np.zeros(b.shape),b,max_it_cg,tol,True)
time_cg = time.time() - t0
print("CG took ",time_cg, " secs")
print("Number of parameters is ",model.count_params())

print("CG test is running - without residual projection")
t0=time.time()
x_sol_cg, res_arr_cg = CG.cg_normal_test(np.zeros(b.shape),b,max_it_cg,tol,True)
time_cg = time.time() - t0
print("CG took ",time_cg, " secs")
print("Number of parameters is ",model.count_params())

res_arr_matrix_name = project_folder_general+"data/res_arr_matrix_cg.npy"
with open(res_arr_matrix_name, 'wb') as f:
    np.save(f, np.array(res_arr_cg))

#%% 




















