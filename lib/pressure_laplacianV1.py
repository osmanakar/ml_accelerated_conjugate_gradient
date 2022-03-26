# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 00:34:19 2021

@author: osman
"""

import numpy as np

class pressure_laplacian:
    def __init__(self, n):
        self.n = n
        self.m = (n+1)**2
        self.A = self.pressure_laplacian_matrix()
        
    # creates pressure laplacian for (n+1)-by-(n+1) matrix
    def pressure_laplacian_matrix(self):
        n_ = self.n+1
        n = self.n
        m = self.m
        A = np.zeros([self.m,self.m])
        for i in range(1,self.n):
            A[i][i]=1
            A[i][n_+i]= -1
            A[m-1-i][m-1-i]=1
            A[m-1-i][m-1-i-n_]=-1
            i0 = n_*i
            A[i0][i0]=1
            A[i0][i0+1]=-1
            i1 = n_*i+n
            A[i1][i1]=1
            A[i1][i1-1]=-1
        for i in range(1,n):
            for j in range(1,n):
                k = n_*i+j
                A[k][k]=4
                A[k][k-1]=-1
                A[k][k+1]=-1
                A[k][k+n_]=-1
                A[k][k-n_]=-1
        return A
    
    # here x is np.array with dimension (n+1)^2
    def multiply_pressure_laplacian(self,x):
        return np.matmul(self.A,x)
        
    # here x is np.array with dimension (n+1)^2
    def multiply_pressure_laplacian_fast(self,x):
        y = np.zeros(self.m)
        n = self.n
        n_ = n+1
        for i in range(1,n):
            y[i] = x[i]-x[n_+i]
            y[self.m-1-i] = x[self.m-1-i] - x[self.m-1-i-n]
            i0 = n_*i
            y[i0] = x[i0] - x[i0+1]
            i1 = n_*i+n
            y[i1] = x[i1] - x[i1-1]
            
        for i in range(1,self.n):
            for j in range(1,self.n):
                k = n_*i+j
                y[k] = 4*x[k] - x[k-1] - x[k+1] - x[k+n_] - x[k-n_]
        return y
    
    def precond(self,x,Q,lambda_):
        y = np.copy(x)
        for i in range(Q.shape[1]):
            qTx = np.dot(Q[:,i],x)
            y = y + qTx*(1/lambda_[i] - 1.0)*Q[:,i]
        return y
            
    
    def create_lambda_vals(self,precond):
        lambda_ = np.ones([precond.shape[1]])
        for i in range(precond.shape[1]):
            dotx = np.dot(precond[:,i],precond[:,i])
            lambda_[i] = np.dot(precond[:,i],self.multiply_pressure_laplacian(precond[:,i]))/dotx
        return lambda_


    def one_step_precond_iteration(self, b, Q):
        #b is rhs
        #x is sol
        #Q is preconditoner, num_eigmodes x m
        w_tilde = np.copy(b)
        lambda_ = self.create_lambda_vals(Q)
        w_bar = self.precond(w_tilde,Q,lambda_)
        mag = np.dot(w_bar,w_tilde)
        if mag<0:
            print(mag)
            mag = 1
        mag = np.sqrt(mag)
        q_bar = w_bar/mag 
        w_tilde = np.matmul(self.A,q_bar)
        alpha = np.dot(q_bar, w_tilde)
        td = mag/alpha
        return q_bar*td

    def res_one_step_precond_iteration(self,b,Q):
        return(np.linalg.norm(np.matmul(self.A,self.one_step_precond_iteration(b,Q)) - b))

    def pcg_normal(self, b, Q, max_it):
        #b is rhs
        #x is sol
        #Q is preconditoner, num_eigmodes x m
        x = np.zeros(b.shape)
        ax = np.matmul(self.A, x)
        #res = np.linalg.norm(ax-b)
        r = b.copy()
        #r = r - ax
        lambda_ = self.create_lambda_vals(Q)
        z = self.precond(b,Q,lambda_)
        p = z.copy()
        rz = np.dot(r,z)
        rz_k = rz;
        for it in range(max_it):
            ax = np.matmul(self.A, p)
            #print(np.dot(p,ax))
            alpha = rz_k/np.dot(p,ax)
            x = x + p*alpha
            r = r - ax*alpha
            z = self.precond(r,Q,lambda_)
            rz = np.dot(r,z)
            beta = rz/rz_k
            pk_1 = p.copy()
            p = z.copy()
            p = p + pk_1*beta
            rz_k = rz
        
        return x
    
    def cg_normal(self,x,b,max_it,tol,verbose=False):
        res = np.linalg.norm(self.multiply_pressure_laplacian(x)-b)
        if verbose:
            print("first cg residual is "+str(res))
        if res < tol:
            if verbose:
                print("CG converged in 0 iterations")
            return x
        ax = self.multiply_pressure_laplacian(x)
        r = b.copy()
        r = r - ax
        p = r.copy()
        rr_k = np.dot(r,r)
        for it in range(max_it):
            ax = self.multiply_pressure_laplacian(p)
            alpha = rr_k/np.dot(p,ax)
            x = x+alpha*p
            res = np.linalg.norm(self.multiply_pressure_laplacian(x)-b)
            if res < tol:
                if verbose:
                    print("CG residual = "+str(res))
                    print("CG converged in "+str(it)+" iteration.")
            r = r - ax*alpha
            rr_k1 = np.dot(r,r)
            beta = rr_k1/rr_k
            q = p.copy()
            p = r.copy()
            p = p+ q*beta
            rr_k = rr_k1
        
        if verbose:
            print("CG residual = " + str(res))
            print("CG used max = "+str(max_it)+" iteration")       
        
    
    
    
    
    
    
