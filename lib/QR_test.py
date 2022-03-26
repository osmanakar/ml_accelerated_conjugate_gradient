# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:08:03 2021

@author: osman
"""
project_folder_general = "C:/Users/osman/Desktop/research_teran/python/MLV1/"
import sys
sys.path.insert(1, project_folder_general+'remote/lib/')
import QR
import numpy as np

Ain=np.array( [[16,1,1,4],[1,12,3,-2],[1,3,-24,2],[4,-2,2,20]]) ## test matrix
B=np.array([len(Ain),len(Ain)])
R=np.identity(len(Ain))
qri =1e-13  
B=Ain
n_step = 0
non_diag_max= QR.max_non_diag_abs_val(B)
while (non_diag_max) >= qri: 
    n_step += 1
    max_index=QR.search_max_index(B)
    p = max_index[0]
    q = max_index[1]
    B,phi = QR.elim_diag(B,p,q)
    R = QR.make_ortho_mat(Ain,R,p,q,phi)

    non_diag_max= QR.max_non_diag_abs_val(B)
    print ("n_step=",n_step,", non_diag_max=",non_diag_max )
#%% 
D = np.matmul(R.transpose(),np.matmul(Ain,R))


#%%

Q = CG.create_ritz_vectors(b_f10,16)
noise_strength = 2
noise_variance = 10
Q_noise = np.random.normal(0,noise_variance, Q.shape)
#print("noise = "+str(np.linalg.norm(Q_noise)))
Q_noise = Q_noise*(noise_strength/np.linalg.norm(Q_noise))
Q_noisy = Q+Q_noise

A_tilde = np.matmul(Q_noisy,np.matmul(pres_lap.A,Q_noisy.transpose()))

R=np.identity(len(A_tilde))
qri =1e-3
B=A_tilde.copy()
n_step = 0
non_diag_max= QR.max_non_diag_abs_val(B)
while (non_diag_max) >= qri: 
    n_step += 1
    max_index=QR.search_max_index(B)
    p = max_index[0]
    q = max_index[1]
    B,phi = QR.elim_diag(B,p,q)
    R = QR.make_ortho_mat(A_tilde,R,p,q,phi)
    non_diag_max= QR.max_non_diag_abs_val(B)
    print ("n_step=",n_step,", non_diag_max=",non_diag_max )


























