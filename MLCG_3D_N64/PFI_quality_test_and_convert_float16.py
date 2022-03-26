project_name = "MLCG_3D_N64"
apple = "/Users/osmanakar/"
windows = "C:/Users/osman/"
#project_folder_general = apple + "OneDrive/research_teran/python/"+project_name+"/"
#project_folder_general = "/home/osman/projects/ML_preconditioner_project/"+project_name+"/"
project_folder_general = "/home/oak/projects/ML_preconditioner_project/"+project_name+"/"
project_data_folder = "/home/oak/projects/ML_preconditioner_project/data/"+project_name+"/"

#%% 
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
import tensorflow as tf 
import gc
import time
import scipy.sparse as sparse
#%%
sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
#import matplotlib.pyplot as plt
import helper_functions as hf

#%%
dim = 64
dim2 = dim**3

#%%
project_folder_subname = sys.argv[1]
epoch_num = int(sys.argv[2])
tol = np.double(sys.argv[3])
b_rhs_n = int(sys.argv[4])
ml_type = sys.argv[5]
 
#%%
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
#pres_lap = pl.pressure_laplacian_3D_sparse(dim-1)
A_sparse = sparse.load_npz(name_sparse_matrix)

#%%
CG = cg.ConjugateGradientSparse(A_sparse)
#%%
#model_3D_N16 = hf.load_model_from_machine(epoch_num, project_folder_general, project_folder_subname,dim, True)
model = hf.load_model_from_source(project_folder_general+project_folder_subname+"/saved_models/"+project_name+"_json_E"+str(epoch_num)+"/")
print(model.summary())
model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual

#%% Getting RHS for the Testing
if b_rhs_n == 0:
    rand_vec_x = np.random.normal(0,1, [dim2])
    b = CG.multiply_A_sparse(rand_vec_x)
else: 
    data_folder_name = project_folder_general+"data/rhs_from_incompressible_flow/output3d_64_from_ayano/"
    n=b_rhs_n
    b = hf.get_frame_from_source(n, data_folder_name, False)
#% %
max_it=100
x_sol, res_arr_ml_generated_cg_3D = CG.cg_on_ML_generated_subspace_test5(b, np.zeros(b.shape), model_predict, max_it,tol, True)

#%%
print("CG test is running")
t0=time.time()
x_sol_cg, res_arr_cg = CG.cg_normal(np.zeros(b.shape),b,1000,tol,True)
time_cg = time.time() - t0
print("CG took ",time_cg, " secs")

#%%
#print("finding the best model")

#%%
d_type_ = tf.float16


if ml_type == "V6":
    fil_num=32
    #fil_dim = 3
    input_rhs = keras.Input(shape=(dim, dim, dim, 1), dtype=d_type_)
    first_layer = layers.Conv3D(16, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)
    for i0 in range(13):
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
        #la = layers.Conv2D(fil_num, (fil_dim, fil_dim), activation='relu', padding='same')(lc) #+ la
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(la)
    ml_model_f16 = keras.Model(input_rhs, last_layer)

elif ml_type == "V20":
    fil_num=32
    #fil_dim = 3
    input_rhs = keras.Input(shape=(dim, dim, dim, 1), dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)
    for i0 in range(16):
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
        #la = layers.Conv2D(fil_num, (fil_dim, fil_dim), activation='relu', padding='same')(lc) #+ la
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(la)
    ml_model_f16 = keras.Model(input_rhs, last_layer)

elif ml_type == "V25":
    fil_num=32
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)
    for i0 in range(5):
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(la)
    ml_model_f16 = keras.Model(input_rhs, last_layer)
    
elif ml_type == "V26":
    fil_num=32
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)
    for i0 in range(3):
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(la)
    ml_model_f16 = keras.Model(input_rhs, last_layer)

elif ml_type == "V25_4":
    fil_num=32
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)
    for i0 in range(5):
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
        #la = layers.Conv2D(fil_num, (fil_dim, fil_dim), activation='relu', padding='same')(lc) + la
    la = layers.Dense(32, activation='linear',dtype=d_type_)(la)
    la = layers.Dense(32, activation='linear',dtype=d_type_)(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(la)
    ml_model_f16 = keras.Model(input_rhs, last_layer)

elif ml_type == "V30_3":
    fil_num=32
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)
    for i0 in range(4):
        lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
        la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
        #la = layers.Conv2D(fil_num, (fil_dim, fil_dim), activation='relu', padding='same')(lc) + la
    la = layers.Dense(32, activation='linear',dtype=d_type_)(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(la)
    ml_model_f16 = keras.Model(input_rhs, last_layer,dtype=d_type_)
    
elif ml_type == "V31_1":
    fil_num=24
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)

    lb = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
    la = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    lb = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
    la = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    lb = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
    la = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    lb = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
    la = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la

    la = layers.Dense(24, activation='linear',dtype=d_type_)(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(la)
    ml_model_f16 = keras.Model(input_rhs, last_layer)

elif ml_type == "V32_1":    
    fil_num=24
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)

    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)

    apa = layers.AveragePooling3D((2, 2,2), padding='same',dtype=d_type_)(lb) #7
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apb) + apa

    upa = layers.UpSampling3D((2, 2,2),dtype=d_type_)(apa) + lb
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(upa) 
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(upb) + upa

    #la = layers.Dense(fil_num, activation='linear')(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(upa)

    ml_model_f16 = keras.Model(input_rhs, last_layer)
    
elif ml_type == "V32_2":    
    fil_num=16
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)

    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)

    apa = layers.AveragePooling3D((2, 2,2), padding='same',dtype=d_type_)(lb) #7
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apb) + apa

    upa = layers.UpSampling3D((2, 2,2),dtype=d_type_)(apa) + lb
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(upa) 
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(upb) + upa

    #la = layers.Dense(fil_num, activation='linear')(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(upa)

    ml_model_f16 = keras.Model(input_rhs, last_layer)

elif ml_type == "V32_4":
    fil_num=16
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (5, 5, 5), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(first_layer)

    lb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(la)
    la = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(lb) + la
    lb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(la)

    apa = layers.AveragePooling3D((2, 2,2), padding='same',dtype=d_type_)(lb) #7
    apb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apb) + apa
    apb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apb) + apa

    upa = layers.UpSampling3D((2, 2,2),dtype=d_type_)(apa) + lb
    upb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(upa) 
    upa = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(upb) + upa
    #la = layers.Dense(fil_num, activation='linear')(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(upa)
    ml_model_f16 = keras.Model(input_rhs, last_layer)


elif ml_type == "V33_1":    
    fil_num=16
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(first_layer)

    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)
    #la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(lb) + la
    #lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(la)

    apa = layers.AveragePooling3D((2, 2,2), padding='same',dtype=d_type_)(lb) #7
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(apb) + apa

    upa = layers.UpSampling3D((2, 2,2),dtype=d_type_)(apa) + lb
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(upa) 
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same',dtype=d_type_)(upb) + upa

    #la = layers.Dense(fil_num, activation='linear')(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(upa)

    ml_model_f16 = keras.Model(input_rhs, last_layer)

elif ml_type == "V37_1":
    fil_num=16
    input_rhs = keras.Input(shape=(dim, dim, dim, 1),dtype=d_type_)
    first_layer = layers.Conv3D(fil_num, (5, 5, 5), activation='linear', padding='same',dtype=d_type_)(input_rhs)
    la = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(first_layer)
    lb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(la)
    #la = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same')(lb) + la
    #lb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same')(la)

    apa = layers.AveragePooling3D((2, 2,2), padding='same',dtype=d_type_)(lb) #7
    apb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apb) + apa
    apb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apa)
    apa = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(apb) + apa

    upa = layers.UpSampling3D((2, 2, 2),dtype=d_type_)(apa) + lb
    upb = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(upa) 
    upa = layers.Conv3D(fil_num, (5, 5, 5), activation='relu', padding='same',dtype=d_type_)(upb) + upa

    #la = layers.Dense(fil_num, activation='linear')(la)
    last_layer = layers.Dense(1, activation='linear',dtype=d_type_)(upa)

    ml_model_f16 = keras.Model(input_rhs, last_layer)


#%%

ml_model_f16.set_weights(model.get_weights())

print("Testing float16 model")
#%%
model_predict_f16 = lambda r: ml_model_f16(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float16),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual

max_it=100
print("Dummy Calling")
x_sol, res_arr_ml_generated_cg = CG.cg_on_ML_generated_subspace_test5(b, np.zeros(b.shape), model_predict_f16, max_it,1e-1, False)

print("Model number of parameters = ",ml_model_f16.count_params())
max_it=100
t0=time.time()
x_sol, res_arr_ml_generated_cg = CG.cg_on_ML_generated_subspace_test5(b, np.zeros(b.shape), model_predict_f16, max_it,tol, True)
time_cg_ml = int(1000*(time.time() - t0))
print("MLCG took ", time_cg_ml, " seconds.")


#%%
print("Saving Float16 Model")
os.system("mkdir ./saved_models_float16/"+project_folder_subname+"_json_float16_E"+str(epoch_num))
os.system("touch ./saved_models_float16/"+project_folder_subname+"_json_float16_E"+str(epoch_num)+"/model.json")
model_json = ml_model_f16.to_json()
model_name_json = project_folder_general+"/saved_models_float16/"+project_folder_subname+"_json_float16_E"+str(epoch_num)+"/"
with open(model_name_json+ "model.json", "w") as json_file:
    json_file.write(model_json)
ml_model_f16.save_weights(model_name_json + "model.h5")

























