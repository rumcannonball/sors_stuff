# -*- coding: utf-8 -*-
"""
NAS Prog Assign 2 - ND, ST, RMcD
"""
import numpy as np

#functions defined here - main method at the bottom
def get_Euclid(x):
    """
    this function accepts one vector
    the calling function should compute this vector as the difference of two vectors
    the standard Euclidean distance formula sqrt(x1^2+x2^2+....+xn^2) is applied
    """
    sum = 0
    for i in x:
        sum+=i**2
    return np.sqrt(sum)
    
def get_residual(A, x, b):
    """
    return the residual error Ax-b for an approximation of x
    input parameters should be numpy arrays
    this is not as simple as using numpy.dot because A is in CSR format 
    """
    adotx = []    
    for i in range(0,len(b)):
        adotx.append(0.0)    
    #i should really do a DOT function instead of coding it explicitly here and in SOR also
    for j in range (0, len(b)): 
        first_nz = A[1][(A[2][j]-1)] - 1 #pos of first non zero on row             
        for k in range(A[2][j]-1, A[2][j+1]-1):            
            adotx[j] = adotx[j] + A[0][k] * x[k - (A[2][j]-1) + first_nz]
    return get_Euclid(np.subtract(adotx, b))    

def get_x_seq(xold, xnew):
    """
    this function computes Euclidean distance between successive iterations of x
    input parameters should be numpy arrays
    """
    return get_Euclid(np.subtract(xnew, xold))

def chk_diverge(xold, xnew, A, b):
    """
    check if previous approx of x was closer than new approx of x
    """
    dist_old = get_residual(A, xold, b)
    dist_new = get_residual(A, xnew, b)
    if dist_old < dist_new:
        return True
    else:
        return False

def chk_converge(A, xnew, b, xold, x_seq_tol, res_tol, flag):
    #checks both residual and x_seq for convergence 
    if flag == True:
        return -1 #required to enter sparse_sor loop
    elif get_residual(A, xnew, b) < res_tol:
        return 2 #dict value for this stopping reason
    elif get_x_seq(xold, xnew) < x_seq_tol:
        return 1
    elif chk_diverge(xold, xnew, A, b) == True:
        return 4
    return -1

def set_output(x, reason, maxits, numits, x_tol, res_tol, filename='nas_Sor.out'):
    """
    save to file
    line 1 - header
    line 2 - stopping reason
    line 3 - computed value for x only if maxits was reached or convergence occurred
    """
    reasons = {
        1 : "x Sequence convergence"
        ,2 : "  Residual convergence"
        ,3 : "Max Iterations reached"
        ,4 : " x Sequence divergence"
        ,5 : "      Zero on diagonal"
        ,6 : "        Cannot proceed"
    }
    
    print('Stopping reason       , Max num of iterations, Number of iterations, Machine epsilon, X seq tolerance, Residual seq tolerance')    
    print(reasons[reason], '            ', maxits, '         ', numits, '            ', np.finfo(float).eps, '       ', x_tol, '        ', res_tol)
    if reason in (1,2,3):
        print(x)
    #leaving alignment and saving file correctly to Srikanth as discussed
    #np.savetxt('nas_Sor.out', res, delimiter = ' ', newline = '\n')    


#toCSR takes a matrix and converts it into dense form
#row1: values, row2: columns, row3: rowstart
def to_csr(mat):
    """
    takes a full line-by-line matrix and compresses into non-zero 'rowstart' CSR format
    row 1 - values, row 2 - columns, row 3 - rowstart
    """
    csr = []
    cols = []
    vals = []
    rowstart = []
    for col in mat:
        col_num = 1
        rowstarted = False
        for val in col:
            if val != 0:
                vals.append(val)
                cols.append(col_num)
                if rowstarted == False:
                    rowstart.append(len(vals))
                    rowstarted = True
            col_num+=1
    rowstart.append(len(vals)+1) #final value
    csr.append(vals)
    csr.append(cols)
    csr.append(rowstart)  
    return csr

def sparse_sor(A, b, n, maxits, x, x_seq_tol, res_tol, w=1.25):
    k = 1
    reason = 6 #something has gone wrong if this does not get overwritten later
    xold = np.array([0.0])
    xnew = np.array(x)
    Anp = np.array(A)
    bnp = np.array(b)
    flag = True #required to enter while loop first time only
    while k <= maxits and chk_converge(Anp, xnew, bnp, xold, x_seq_tol, res_tol, flag) == -1:
        flag = False        
        xold = np.array(xnew[:])
        for i in range(0,len(b)):
            sum = 0
            first_nz = Anp[1][(Anp[2][i]-1)] - 1 #pos of first non zero on row
            for j in range(Anp[2][i]-1, Anp[2][i+1]-1):
                sum = sum + Anp[0][j] * xnew[j - (Anp[2][i]-1) + first_nz]
                if Anp[1][j] == i+1:
                    d = A[0][j]
                    if d== 0:
                        reason=5
            xnew[i] = xnew[i] + w * (bnp[i] - sum) / d   
        k+=1
    conv = chk_converge(Anp, xnew, bnp, xold, x_seq_tol, res_tol, False)
    if k-1 == maxits:
        reason = 3
    elif conv != -1:
        reason = conv
    set_output(xnew, reason, maxits, k-1, x_seq_tol, res_tol)

#MAIN method starts here
#read in values for n, A, b
n = int(open('nas_Sor.in', 'r').readline())
A_mat = np.loadtxt('nas_Sor.in',delimiter=' ',skiprows=1)
b_vec = A_mat[n]
A_mat = np.delete(A_mat, (n), axis=0) #remove vector b from matrix A_mat

#initial guess of x is all zeros
x = []
for i in range(0,n):
    x.append(0.0)
A = to_csr(A_mat)
print(A_mat)
print(b_vec)
print(n)
print(A)

#solve linear system
sparse_sor(A, b_vec, n, 50, x, 0.00000000000001, 0.00000000000001)
