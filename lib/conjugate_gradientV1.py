# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 00:34:19 2021

@author: osman
"""

import numpy as np

class ConjugateGradient:
    def __init__(self, A):
        if A.shape[0] != A.shape[1]:
            print("A is not a square matrix!")
        self.n = A.shape[0]
        self.A = A.copy()
        self.machine_tol = 1.0e-17;
        
    
    # here x is np.array with dimension n
    def multiply_A(self,x):
        return np.matmul(self.A,x)
    
    def create_approximate_eigenmodes(self, b, num_modes, sorting=True):
        W, diagonal, sub_diagonal = self.lanczos_iteration(b, num_modes, 1.0e-10)
        if(num_modes != len(diagonal)):
            print("Careful. Lanczos Iteration converged too early, num_modes = "+str(num_modes)+" > "+str(len(diagonal)))
            num_modes = len(diagonal)
        tri_diag = np.zeros([num_modes,num_modes])
        for i in range(1,num_modes-1):
            tri_diag[i,i] = diagonal[i]
            tri_diag[i,i+1] = sub_diagonal[i]
            tri_diag[i,i-1]= sub_diagonal[i-1]
        tri_diag[0,0]=diagonal[0]
        tri_diag[0,1]=sub_diagonal[0]
        tri_diag[num_modes-1,num_modes-1]=diagonal[num_modes-1]
        tri_diag[num_modes-1,num_modes-2]=sub_diagonal[num_modes-2]
        eigvals,Q0 = np.linalg.eig(tri_diag)
        Q1 = np.matmul(W.transpose(),Q0).transpose()
        if sorting:
            Q = np.zeros([num_modes,self.n])
            sorted_eig_vals = sorted(range(num_modes), key=lambda k: -eigvals[k])
            for i in range(num_modes):
                Q[i]=Q1[sorted_eig_vals[i]].copy()
            return Q
        else:
            return Q1
        
    def create_approximate_eigenmodes_with_ray_quo_iter(self, b, num_modes, lambda_vals, initial_guess, tol=1.0e-10, sorting=True, verbose = False):
        W, diagonal, sub_diagonal = self.lanczos_iteration(b, num_modes, 1.0e-10)
        initial_guess0 = np.matmul(W, initial_guess.transpose())
        print(initial_guess0.shape)
        if(num_modes != len(diagonal)):
            print("Error! Lanczos Iteration converged too early, num_modes = "+str(num_modes)+" > "+str(len(diagonal)))
            return
        if(num_modes != len(lambda_vals)):
            print("Error!. num_modes != len(lambda_vals)")
            return
        
        tri_diag = np.zeros([num_modes,num_modes])
        for i in range(1,num_modes-1):
            tri_diag[i,i] = diagonal[i]
            tri_diag[i,i+1] = sub_diagonal[i]
            tri_diag[i,i-1]= sub_diagonal[i-1]
        tri_diag[0,0]=diagonal[0]
        tri_diag[0,1]=sub_diagonal[0]
        tri_diag[num_modes-1,num_modes-1]=diagonal[num_modes-1]
        tri_diag[num_modes-1,num_modes-2]=sub_diagonal[num_modes-2]
        
        Q_tilde = np.zeros([num_modes,num_modes])
        new_lambda_vals = np.zeros(num_modes)
        #Applying Rayleigh Quotient Iteration
        for k in range(num_modes):
            #print("k = " +str(k))
            mu = lambda_vals[k]
            #v0 = np.random.rand(b.shape[0])
            v0 = initial_guess0[k]
            for j in range(30): #applying inverse power iteration
                w = np.linalg.solve(tri_diag-mu*np.identity(num_modes),v0)
                v1 = w/np.linalg.norm(w)
                mu = np.dot(v1,np.matmul(tri_diag,v1))                    
                v0 = v1.copy()
                err = np.linalg.norm(mu*v0-np.matmul(tri_diag,v0))
                if verbose:
                    print("Inside CG: eigmode k = "+str(k) + ", IncPowIt j = "+str(j) + " -> "+str(mu))
                    print("||Av - lambda*v||_2 = "+str(err))
                if err<tol: 
                    Q_tilde[k]=v0.copy()
                    new_lambda_vals[k]=mu
                    break
            Q_tilde[k]=v0.copy()
            new_lambda_vals[k]=mu
        aa = np.matmul(Q_tilde,np.matmul(tri_diag,Q_tilde.transpose()))
        Q = np.matmul(W.transpose(),Q_tilde).transpose()
        return Q, new_lambda_vals, aa
        
        
    #Q is num_modes x self.n np array
    def mult_precond_approximate_eigenmodes(self,x,Q,lambda_):
        y = np.copy(x)
        for i in range(Q.shape[0]):
            qTx = np.dot(Q[i],x)
            y = y + qTx*(1/lambda_[i] - 1.0)*Q[i]
        return y
            
    #precond is num_modes x self.n np array
    def create_lambda_vals(self,precond):
        lambda_ = np.zeros(precond.shape[0])
        for i in range(precond.shape[0]):
            dotx = np.dot(precond[i],precond[i])
            if dotx < self.machine_tol:
                print("Error! Zero vector in Preconditioner.")
                return
            lambda_[i] = np.dot(precond[i],np.matmul(self.A,precond[i]))/dotx
        return lambda_

    def mult_diag_precond(self,x):
        y = np.zeros(x.shape)
        for i in range(self.n):
            if self.A[i,i]>self.machine_tol:
                y[i] = x[i]/self.A[i,i]
        return y
    
 
    def pcg_normal_with_diagonal_preconditioner(self, x_init, b, epsilon, max_it=100, tol=1.0e-12,verbose=True):
        #b is rhs
        #x_init is initial prediction
        #Q is preconditoner, num_eigmodes x m
        res_arr = []
        x = x_init.copy()
        ax = np.matmul(self.A, x_init)
        res = np.linalg.norm(ax-b)
        res_arr = res_arr + [res]
        if verbose:
            print("First PCG residual = "+str(res))
        if res<tol:
            if verbose:
                print("PCG converged in 0 iteration. Final residual is "+str(res))
            return [x, res_arr]
        
        r = b.copy()
        r = r - ax
        z = epsilon*self.mult_diag_precond(r)
        p = z.copy()
        rz = np.dot(r,z)
        rz_k = rz;
        for it in range(max_it):
            ax = np.matmul(self.A, p)
            alpha = rz_k/np.dot(p,ax)
            x = x + p*alpha
            r = r - ax*alpha
            z = epsilon*self.mult_diag_precond(r)
            rz = np.dot(r,z)
            res = np.linalg.norm(np.matmul(self.A, x)-b)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("PCG residual = "+str(res))
                    print("PCG converged in "+str(it)+ " iterations.")
                return [x, res_arr]
            beta = rz/rz_k
            pk_1 = p.copy()
            p = z.copy()
            p = p + pk_1*beta
            rz_k = rz
        
        if verbose:
            print("PCG residual = "+str(res))
            print("PCG converged in "+str(max_it)+ " iterations.")
            
        return [x, res_arr]
    

    def pcg_normal(self, x_init, b, Q, max_it=100, tol=1.0e-12,verbose=True):
        #b is rhs
        #x_init is initial prediction
        #Q is preconditoner, num_eigmodes x m
        res_arr = []
        x = x_init.copy()
        ax = np.matmul(self.A, x_init)
        res = np.linalg.norm(ax-b)
        res_arr = res_arr + [res]
        if verbose:
            print("First PCG residual = "+str(res))
        if res<tol:
            if verbose:
                print("PCG converged in 0 iteration. Final residual is "+str(res))
            return [x, res_arr]
        
        r = b.copy()
        r = r - ax
        lambda_ = self.create_lambda_vals(Q)
        z = self.mult_precond_approximate_eigenmodes(r,Q,lambda_)
        p = z.copy()
        rz = np.dot(r,z)
        rz_k = rz;
        for it in range(max_it):
            ax = np.matmul(self.A, p)
            alpha = rz_k/np.dot(p,ax)
            x = x + p*alpha
            r = r - ax*alpha
            z = self.mult_precond_approximate_eigenmodes(r,Q,lambda_)
            rz = np.dot(r,z)
            res = np.linalg.norm(np.matmul(self.A, x)-b)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("PCG residual = "+str(res))
                    print("PCG converged in "+str(it)+ " iterations.")
                return [x, res_arr]
            beta = rz/rz_k
            pk_1 = p.copy()
            p = z.copy()
            p = p + pk_1*beta
            rz_k = rz
        
        if verbose:
            print("PCG residual = "+str(res))
            print("PCG converged in "+str(max_it)+ " iterations.")
            
        return [x, res_arr]
    
    def cg_normal(self,x,b,max_it,tol,verbose=False):
        res = np.linalg.norm(self.multiply_A(x)-b)
        res_arr = [res]
        if verbose:
            print("first cg residual is "+str(res))
        if res < tol:
            if verbose:
                print("CG converged in 0 iterations")
            return [x, res_arr]
        ax = self.multiply_A(x)
        r = b.copy()
        r = r - ax
        p = r.copy()
        rr_k = np.dot(r,r)
        for it in range(max_it):
            ax = self.multiply_A(p)
            alpha = rr_k/np.dot(p,ax)
            x = x+alpha*p
            res = np.linalg.norm(self.multiply_A(x)-b)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("CG residual = "+str(res))
                    print("CG converged in "+str(it)+" iteration.")
                return [x, res_arr]
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
        
        return [x, res_arr]
    
   
    def pcg_normalV2(self, x_init, b, Q, epsilon=0.1, max_it=100, tol=1.0e-12,verbose=True):
        #b is rhs
        #x_init is initial prediction
        #Q is preconditoner, num_eigmodes x m
        res_arr = []
        x = x_init.copy()
        ax = np.matmul(self.A, x_init)
        res = np.linalg.norm(ax-b)
        res_arr = res_arr + [res]
        if verbose:
            print("First PCG residual = "+str(res))
        if res<tol:
            if verbose:
                print("PCG converged in 0 iteration. Final residual is "+str(res))
            return [x, res_arr]
        
        r = b.copy()
        r = r - ax
        lambda_ = self.create_lambda_vals(Q)
        z = self.mult_precond_approximate_eigenmodes(r,Q,lambda_) + epsilon*self.mult_diag_precond(r)
        #z = epsilon*self.mult_diag_precond(r)
        p = z.copy()
        rz = np.dot(r,z)
        rz_k = rz;
        for it in range(max_it):
            ax = np.matmul(self.A, p)
            alpha = rz_k/np.dot(p,ax)
            x = x + p*alpha
            r = r - ax*alpha
            z = self.mult_precond_approximate_eigenmodes(r,Q,lambda_) + epsilon*self.mult_diag_precond(r)
            #z = epsilon*self.mult_diag_precond(r)
            rz = np.dot(r,z)
            res = np.linalg.norm(np.matmul(self.A, x)-b)
            res_arr = res_arr + [res]
            if res < tol:
                if verbose:
                    print("PCG residual = "+str(res))
                    print("PCG converged in "+str(it)+ " iterations.")
                return [x, res_arr]
            beta = rz/rz_k
            pk_1 = p.copy()
            p = z.copy()
            p = p + pk_1*beta
            rz_k = rz
        
        if verbose:
            print("PCG residual = "+str(res))
            print("PCG converged in "+str(max_it)+ " iterations.")
            
        return [x, res_arr]
    
 
    
    def lanczos_iteration(self, b, max_it=10, tol=1.0e-10):
        if max_it > self.n:
            max_it = self.n
        diagonal = np.zeros(max_it)
        sub_diagonal = np.zeros(max_it)
        Q = np.zeros([max_it, self.n])
        norm_b = np.linalg.norm(b)
        Q[0] = b.copy()/norm_b
        Q[1] = np.matmul(self.A, Q[0])
        diagonal[0] = np.dot(Q[1],Q[0])
        Q[1] = Q[1] - diagonal[0]*Q[0]
        sub_diagonal[0] = np.linalg.norm(Q[1])
        Q[1] = Q[1]/sub_diagonal[0]
        if sub_diagonal[0]<tol:
            Q = np.resize(Q,[1,self.n])
            diagonal = np.resize(diagonal, [1])
            sub_diagonal = np.resize(sub_diagonal, [0])
            return Q, diagonal, sub_diagonal
        
        invariant_subspace = False
        it = 1
        while ((it<max_it-1) and (not invariant_subspace)):
            Q[it+1] = np.matmul(self.A, Q[it])
            diagonal[it] = np.dot(Q[it],Q[it+1])
            Q[it+1] = Q[it+1] - diagonal[it]*Q[it]-sub_diagonal[it-1]*Q[it-1]
            sub_diagonal[it] = np.linalg.norm(Q[it+1])
            Q[it+1] = Q[it+1]/sub_diagonal[it]
            if sub_diagonal[it] < tol:
                invariant_subspace = True
            it = it+1
            
        Q = np.resize(Q, [it+1,self.n])
        diagonal = np.resize(diagonal, [it+1])
        sub_diagonal = np.resize(sub_diagonal, [it])
        if not invariant_subspace:
            diagonal[it] = np.dot(Q[it], np.matmul(self.A,Q[it]))
        
        return Q, diagonal, sub_diagonal
                
    def restarted_pcg(self, b, max_outer_it = 100, pcg_inner_it = 1, tol = 1.0e-15, method = "approximate_eigenmodes", num_modes = 16, verbose = False):
        res_arr = []
        x_sol = np.zeros(b.shape)
        b_iter = b.copy()
        x_init = np.zeros(b.shape)
        for i in range(max_outer_it):
            if method == "approximate_eigenmodes":
                Q = self.create_approximate_eigenmodes(b_iter,num_modes)
            else:
                print("Method is not recognized!")
                return
            x_sol1, res_arr1 = self.pcg_normal(x_init, b_iter, Q, pcg_inner_it, tol, False)
            x_sol = x_sol + x_sol1
            b_iter = b - np.matmul(self.A,x_sol)
            b_norm = np.linalg.norm(b_iter)
            res_arr = res_arr + res_arr1[0:pcg_inner_it]

            if verbose:
                print("restarting at i = "+ str(i)+ " , residual = "+ str(b_norm))            
            if b_norm < tol:
                print("RestartedPCG converged in "+str(i)+" iterations.")
                break
        return x_sol, res_arr
    
    
    
    
    
