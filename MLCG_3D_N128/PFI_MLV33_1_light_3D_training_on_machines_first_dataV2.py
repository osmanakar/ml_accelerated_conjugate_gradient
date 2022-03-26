import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf 
import gc
import scipy.sparse as sparse
#import matplotlib.pyplot as plt



project_name = "MLCG_3D_N128"
project_folder_subname = os.path.basename(os.getcwd())
print("project_folder_subname = ", project_folder_subname)
project_folder_general = "/home/oak/projects/ML_preconditioner_project/"+project_name+"/"
project_data_folder = "/home/oak/projects/ML_preconditioner_project/data/"+project_name+"/"


sys.path.insert(1, project_folder_general+'../lib/')
import conjugate_gradient as cg
import pressure_laplacian as pl
import helper_functions as hf

dim = 128
dim2 = dim**3
#%%
epoch_num = int(sys.argv[1])
epoch_each_iter = 1 #int(sys.argv[2])
gpu_usage = int(1024*np.double(sys.argv[2]))
b_size = int(sys.argv[3])
#lr = np.double(sys.argv[5])
lr = 1.0e-4

"""
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_usage)])
  except RuntimeError as e:
    print(e)
"""
# Using the second gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

#%% Creating ConjugateGradientSparse Object
print("Creating ConjugateGradientSparse Object")

#pres_lap_sparse = pl.pressure_laplacian_sparse(dim-1)
#pres_lap = pl.pressure_laplacian(dim-1)
name_sparse_matrix = project_folder_general+"data/A_Sparse_3D_N"+str(dim-1)+".npz"
#sparse.save_npz(name_sparse_matrix, pres_lap_sparse.A_sparse)
A_sparse = sparse.load_npz(name_sparse_matrix)
CG = cg.ConjugateGradientSparse(A_sparse)

with open(project_data_folder+'test_idx_10000'+'.npy', 'rb') as f:
    test_idx = np.load(f)
with open(project_data_folder+'train_idx_10000'+'.npy', 'rb') as f:
    train_idx = np.load(f)


d_name = "b_rhs_10000_2_10000_faulty_ritz_vectors_V2_for_3D_random_N"                
with open(project_data_folder+d_name+str(dim-1)+'.npy', 'rb') as f:  
    b_rhs = np.load(f) #150~160 GB        
b_rhs = b_rhs.reshape([len(b_rhs),dim,dim,dim,1]) #140GB

d_name = "b_rhs_10000_2_10000_faulty_ritz_vectors_V2_for_3D_reshaped_random_N"                
with open(project_data_folder+d_name+str(dim-1)+'.npy', 'wb') as f:  
    np.save(f,b_rhs) #150~160 GB   



#%%
coo = A_sparse.tocoo()
indices = np.mat([coo.row, coo.col]).transpose()
A_sparse = tf.SparseTensor(indices, np.float32(coo.data), coo.shape)
#data_folder_name = project_folder_general+"data/rhs_from_incompressible_flow/"
#b = hf.get_frame_from_source(300, data_folder_name)

#%%    
def custom_loss_function_cnn_1d_fast(y_true,y_pred):
    b_size_ = len(y_true)
    err = 0
    for i in range(b_size):
        A_tilde_inv = 1/tf.tensordot(tf.reshape(y_pred[i],[1,dim2]), tf.sparse.sparse_dense_matmul(A_sparse, tf.reshape(y_pred[i],[dim2,1])),axes=1)
        qTb = tf.tensordot(tf.reshape(y_pred[i],[1,dim2]), tf.reshape(y_true[i],[dim2,1]), axes=1)
        x_initial_guesses = tf.reshape(y_pred[i],[dim2,1]) * qTb * A_tilde_inv
        err = err + tf.reduce_sum(tf.math.square(tf.reshape(y_true[i],[dim2,1]) - tf.sparse.sparse_dense_matmul(A_sparse, x_initial_guesses)))
    return err/b_size_

#%%
fil_num=16
input_rhs = keras.Input(shape=(dim, dim, dim, 1))
first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same')(input_rhs)
la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(first_layer)
lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
#la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
#lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)

apa = layers.AveragePooling3D((2, 2,2), padding='same')(lb) #7
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa

upa = layers.UpSampling3D((2, 2,2))(apa) + lb
upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa) 
upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa

#la = layers.Dense(fil_num, activation='linear')(la)
last_layer = layers.Dense(1, activation='linear')(upa)

model = keras.Model(input_rhs, last_layer)
#model_PFI_V1.compile(optimizer="Adam", loss='MeanSquaredError') #MeanSquaredError, MeanSquaredLogarithmicError
model.compile(optimizer="Adam", loss=custom_loss_function_cnn_1d_fast) #MeanSquaredError, MeanSquaredLogarithmicError
model.optimizer.lr = lr;
model.summary()
#%%
rand_vec_x = np.random.normal(0,1, [dim2])
b = CG.multiply_A_sparse(rand_vec_x)
data_folder_name = project_folder_general+"data/rhs_from_incompressible_flow/output3d_128_from_ayano/"
n=10
b = hf.get_frame_from_source(n, data_folder_name)


#%%
training_loss_name = project_folder_general+project_folder_subname+"/"+project_name+"_training_loss.npy"
validation_loss_name = project_folder_general+project_folder_subname+"/"+project_name+"_validation_loss.npy"
training_loss = []
validation_loss = []
for i in range(1,epoch_num):
    print("Training at i = " + str(i))
    ii = (i-1)%4
    if ii == 0:
        train_idx0 = train_idx[0:4500]
        test_idx0 = test_idx[0:500]
        d_name = "b_rhs_10000_1_10000_faulty_ritz_vectors_V2_for_3D_random_N"                
        with open(project_data_folder+d_name+str(dim-1)+'.npy', 'rb') as f:  
            b_rhs = np.load(f) #150~160 GB        
        x_train = tf.convert_to_tensor(b_rhs[train_idx].reshape([len(train_idx),dim,dim,dim,1]),dtype=tf.float32) #140GB
        x_test = tf.convert_to_tensor(b_rhs[test_idx].reshape([len(test_idx),dim,dim,dim,1]),dtype=tf.float32) #16GB
        del b_rhs
        gc.collect(generation=2)

    if ii == 1:
        train_idx0 = train_idx[4500:9000]
        test_idx0 = test_idx[500:1000]
        d_name = "b_rhs_10000_1_10000_faulty_ritz_vectors_V2_for_3D_random_N"                
        with open(project_data_folder+d_name+str(dim-1)+'.npy', 'rb') as f:  
            b_rhs = np.load(f) #150~160 GB 
        x_train = tf.convert_to_tensor(b_rhs[train_idx].reshape([len(train_idx0),dim,dim,dim,1]),dtype=tf.float32) #140GB
        x_test = tf.convert_to_tensor(b_rhs[test_idx].reshape([len(test_idx0),dim,dim,dim,1]),dtype=tf.float32) #16GB
        del b_rhs
        gc.collect(generation=2)

    if ii == 2:
        train_idx0 = train_idx[0:4500]
        test_idx0 = test_idx[0:500]
        d_name = "b_rhs_10000_2_10000_faulty_ritz_vectors_V2_for_3D_random_N"                
        with open(project_data_folder+d_name+str(dim-1)+'.npy', 'rb') as f:  
            b_rhs = np.load(f) #150~160 GB        
        x_train = tf.convert_to_tensor(b_rhs[train_idx].reshape([len(train_idx0),dim,dim,dim,1]),dtype=tf.float32) #140GB
        x_test = tf.convert_to_tensor(b_rhs[test_idx].reshape([len(test_idx0),dim,dim,dim,1]),dtype=tf.float32) #16GB

    if ii == 3:
        train_idx0 = train_idx[4500:9000]
        test_idx0 = test_idx[500:1000]
        d_name = "b_rhs_10000_2_10000_faulty_ritz_vectors_V2_for_3D_random_N"                
        with open(project_data_folder+d_name+str(dim-1)+'.npy', 'rb') as f:  
            b_rhs = np.load(f) #150~160 GB        
        x_train = tf.convert_to_tensor(b_rhs[train_idx].reshape([len(train_idx0),dim,dim,dim,1]),dtype=tf.float32) #140GB
        x_test = tf.convert_to_tensor(b_rhs[test_idx].reshape([len(test_idx0),dim,dim,dim,1]),dtype=tf.float32) #16GB        
        del b_rhs
        gc.collect(generation=2)
    
    hist = model.fit(x_train,x_train,
                    epochs=epoch_each_iter,
                    batch_size=b_size,
                    shuffle=True,
                    validation_data=(x_test,x_test))
    del x_train, x_test
    gc.collect(generation=2)
    
    training_loss = training_loss + hist.history['loss']
    validation_loss = validation_loss + hist.history['val_loss']    
    os.system("mkdir ./saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i))
    os.system("touch ./saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/model.json")
    model_json = model.to_json()
    model_name_json = project_folder_general+project_folder_subname+"/saved_models/"+project_name+"_json_E"+str(epoch_each_iter*i)+"/"
    with open(model_name_json+ "model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(model_name_json + "model.h5")
    
    with open(training_loss_name, 'wb') as f:
        np.save(f, np.array(training_loss))
    with open(validation_loss_name, 'wb') as f:
        np.save(f, np.array(validation_loss))
    print(training_loss)
    print(validation_loss)
    
    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1,dim,dim,dim]),dtype=tf.float32),training=False).numpy()[0,:,:].reshape([dim2]) #first_residual
    max_it=100
    tol=1.0e-12
    x_sol, res_arr_ml_generated_cg = CG.cg_on_ML_generated_subspace_test5(b, np.zeros(b.shape), model_predict, max_it,tol, True)



