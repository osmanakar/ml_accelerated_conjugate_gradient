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
epoch_num_start = int(sys.argv[3])
epoch_num_finish = int(sys.argv[4])

#%% Getting the matrix
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
#pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = sparse.load_npz(name_sparse_matrix)
#%%
CG = cg.ConjugateGradientSparse(A_sparse)
#%% Getting RHS for the Testing
data_folder_name = project_folder_general+"data/rhs_from_incompressible_flow/output3d_64_from_ayano/"
n=10
b = hf.get_frame_from_source(n, data_folder_name)
#%% 
max_it=100
res_arr_matrix = np.zeros([epoch_num_finish,max_it+1])

zero_rows = CG.get_zero_rows()
zero_vectors = np.ones([1,dim2])
zero_vectors[0,zero_rows] = 0.0
zero_vectors[0] = zero_vectors[0]/np.linalg.norm(zero_vectors[0])


for epoch_num in range(epoch_num_start,epoch_num_finish):
    print("epoch_num = ",epoch_num)
    model = hf.load_model_from_source(project_folder_general+project_folder_subname+"/saved_models/"+project_name+"_json_E"+str(epoch_num)+"/")
    print("model has parameters is ",model.count_params())
    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual
    x_sol, res_arr = CG.cg_on_ML_generated_subspace_test4(b, np.zeros(b.shape), model_predict, max_it,tol, True, zero_vectors)    
    res_arr_matrix[epoch_num, 0:len(res_arr)]=np.array(res_arr)

#%%
#print("CG test is running")
#x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,1000,tol,True)
#print("Number of parameters is ",model.count_params())

#%%
res_arr_matrix_name = project_folder_general+"data/test_picking_best_model/res_arr_matrix_"+project_folder_subname+str(epoch_num_start)+"_"+str(epoch_num_finish)+".npy"
with open(res_arr_matrix_name, 'wb') as f:
    np.save(f, np.array(res_arr_matrix))



















