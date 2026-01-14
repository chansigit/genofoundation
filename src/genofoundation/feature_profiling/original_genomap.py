"""
Gromov-Wasserstein optimal transport method for Genomap construction
===================================
author: anonymous

"""
# Adjusted from:
#
# Author: Erwan Vautier <erwan.vautier@gmail.com>
#         Nicolas Courty <ncourty@irisa.fr>
#
# License: MIT License

import numpy as np
import ot


def tensor_square_loss_adjusted(C1, C2, T):
    r"""
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the square loss
    function as the loss function of Gromow-Wasserstein discrepancy.

    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        T : A coupling between those two spaces

    The square-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            f1(a)=(a^2)/2
            f2(b)=(b^2)/2
            h1(a)=a
            h2(b)=b

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T : ndarray, shape (ns, nt)
         Coupling between source and target spaces

    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a**2) / 2

    def f2(b):
        return (b**2) / 2

    def h1(a):
        return a

    def h2(b):
        return b

    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()

    return tens


def tensor_KL_loss_adjusted(C1, C2, T):
    r"""
    Returns the value of \mathcal{L}(C1,C2) \otimes T with the KL loss
    function as the loss function of Gromow-Wasserstein discrepancy.

    Where :
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        T : A coupling between those two spaces

    The KL-loss function L(a,b)=(1/2)*|a-b|^2 is read as :
        L(a,b) = f1(a)+f2(b)-h1(a)*h2(b) with :
            f1(a)=alog(a)-a
            f2(b)=b
            h1(a)=a
            h2(b)=log(b)

    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    T : ndarray, shape (ns, nt)
         Coupling between source and target spaces

    Returns
    -------
    tens : ndarray, shape (ns, nt)
           \mathcal{L}(C1,C2) \otimes T tensor-matrix multiplication result
    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    def f1(a):
        return (a*np.log(a+1e-15)-a)

    def f2(b):
        return b

    def h1(a):
        return a

    def h2(b):
        return (np.log(b+1e-15))

    tens = -np.dot(h1(C1), T).dot(h2(C2).T) 
    tens -= tens.min()

    return tens

def create_space_distributions(num_locations, num_cells):
    """Creates uniform distributions at the target and source spaces.
    num_locations -- the number of locations at the target space
    num_cells     -- the number of single-cells in the data."""
    p_locations = ot.unif(num_locations)
    p_expression = ot.unif(num_cells)
    return p_locations, p_expression

def compute_random_coupling(p, q, epsilon):
    r"""
    Computes a random coupling based on:

    KL-Proj_p,q(K) = argmin_T <-\epsilon logK, T> -\epsilon H(T)
    where T is a couping matrix with marginal distributions p, and q, for rows and columns, respectively

    This is solved with a Bregman Sinkhorn computation
    p       -- marginal distribution of rows
    q       -- marginal distribution of columns
    epsilon -- entropy coefficient
    """
    num_cells = len(p)
    num_locations = len(q)
    K = np.random.rand(num_cells, num_locations)
    C = -epsilon * np.log(K)
    return sinkhorn(p, q, C, epsilon,method='sinkhorn')

def gromov_wasserstein_adjusted_norm(cost_mat, C1, C2,p, q, loss_fun, epsilon,
                                     max_iter=1000, tol=1e-9, verbose=False, log=False, random_ini=False):
    r"""
    Returns the gromov-wasserstein coupling between the two measured similarity matrices

    (C1,p) and (C2,q)

    The function solves the following optimization problem:

    .. math::
        \GW = arg\min_T \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))

        s.t. \GW 1 = p

             \GW^T 1= q

             \GW\geq 0

    Where :
        M  : cost matrix in sourceXtarget space
        C1 : Metric cost matrix in the source space
        C2 : Metric cost matrix in the target space
        p  : distribution in the source space
        q  : distribution in the target space
        L  : loss function to account for the misfit between the similarity matrices
        H  : entropy

    Parameters
    ----------
    M : ndarray, shape (ns, nt)
         Cost matrix in the sourceXtarget space
    C1 : ndarray, shape (ns, ns)
         Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
         Metric costfr matrix in the target space
    p :  ndarray, shape (ns,)
         distribution in the source space
    q :  ndarray, shape (nt,)
         distribution in the target space
    loss_fun :  string
        loss function used for the solver either 'square_loss' or 'kl_loss'
    epsilon : float
        Regularization term >0
    max_iter : int, optional
       Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        record log if True

    Returns
    -------
    T : ndarray, shape (ns, nt)
        coupling between the two spaces that minimizes :
            \sum_{i,j,k,l} L(C1_{i,k},C2_{j,l})*T_{i,j}*T_{k,l}-\epsilon(H(T))
    """

    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)
    cost_mat = np.asarray(cost_mat, dtype=np.float64)

    T = compute_random_coupling(p, q, epsilon) if random_ini else np.outer(p, q)  # Initialization

    cpt = 0
    err = 1

      
    while (err > tol and cpt < max_iter):
        Tprev = T

        if loss_fun == 'square_loss':
            tens = tensor_square_loss_adjusted(C1, C2, T)
        if loss_fun == 'kl_loss':
                tens = tensor_KL_loss_adjusted(C1, C2, T)

        
        if epsilon ==0:
            T= ot.lp.emd(p, q, tens)
        else:
            T = sinkhorn(p, q, tens, epsilon,numItermax=max_iter)
    
        if cpt % 10 == 0:
        # We can speed up the process by checking for the error only all
        # the 10th iterations
            err = np.linalg.norm(T - Tprev)

            if log:
                log['err'].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print('{:5s}|{:12s}'.format(
                            'It.', 'Err') + '\n' + '-' * 19)
                    print('{:5d}|{:8e}|'.format(cpt, err))

        cpt += 1

    if log:
        return T, log
    else:
        return T


import numpy as np
import sklearn.metrics as mpd
#from genomap.genomapOPT import create_space_distributions, gromov_wasserstein_adjusted_norm

def createMeshDistance(rowNum,colNum):
    """
    Returns the Euclidean distance matrix in genomap space
    
    Where :
        rowNum : number of rows in a genomap
        colNum : number of columns in a genomap

    Parameters
    ----------
    rowNum : int, 
         number of rows in a genomap
    colNum : int,
         number of columns in a genomap

    Returns
    -------
    distMat : ndarray, shape (rowNum*colNum, rowNum*colNum)
    """

    # If the row number is even
    if (rowNum % 2) == 0:
        Nx=rowNum/2
        x = np.linspace(-Nx, Nx-1, rowNum)
    # If the row number is odd
    else:
        Nx=(rowNum-1)/2
        x = np.linspace(-Nx, Nx, rowNum)

    # If the column number is even
    if (colNum % 2) == 0:
        Mx=colNum/2
        y = np.linspace(-Mx, Mx-1, colNum)
    # If the column number is odd
    else:
       Mx=(colNum-1)/2
       y = np.linspace(-Mx, Mx, colNum)

    # Create 2D mesh grid from 1D x and y grids
    xx, yy = np.meshgrid(x, y)
    # Compute Euclidean distance between grid points
    zz = np.sqrt(xx**2 + yy**2)
    # Make the 2D grid into a 1D vector and form the Euclidean distance matrix
    gridVec=zz.flatten()
    distMat=mpd.pairwise_distances(gridVec.reshape(-1,1))
    return distMat


def createInteractionMatrix(data, metric='correlation'):
    """
    Returns the interaction matrix among the genes

    Parameters
    ----------
    data : ndarray, shape (cellNum, geneNum)
         gene expression data in cell X gene format. Each row corresponds
         to one cell, whereas each column represents one gene
    metric : 'string'
         Metric for computing the genetic interaction

    Returns
    -------
    interactMat : ndarray, shape (geneNum, geneNum)
           pairwise interaction matrix among genes
    """

    interactMat=mpd.pairwise_distances(data.T,metric=metric)
    return interactMat


def construct_genomap(data,rowNum,colNum,epsilon=0,num_iter=1000):
    """
    Returns the constructed genomaps


    Parameters
    ----------
    data : ndarray, shape (cellNum, geneNum)
         gene expression data in cell X gene format. Each row corresponds
         to one cell, whereas each column represents one gene
    rowNum : int, 
         number of rows in a genomap
    colNum : int,
         number of columns in a genomap

    Returns
    -------
    genomaps : ndarray, shape (rowNum, colNum, zAxisDimension, cell number)
           genomaps are formed for each cell. zAxisDimension is more than
           1 when 3D genomaps are created. 
    """

    sizeData=data.shape
    numCell=sizeData[0]
    numGene=sizeData[1]
    # distance matrix of 2D genomap grid
    distMat = createMeshDistance(rowNum,colNum)
    # gene-gene interaction matrix 
    interactMat = createInteractionMatrix(data, metric='correlation')

    totalGridPoint=rowNum*colNum
    
    if (numGene<totalGridPoint):
        totalGridPointEff=numGene
    else:
        totalGridPointEff=totalGridPoint
    
    M = np.zeros((totalGridPointEff, totalGridPointEff))
    p, q = create_space_distributions(totalGridPointEff, totalGridPointEff)

   # Coupling matrix 
    T = gromov_wasserstein_adjusted_norm(
    M, interactMat, distMat[:totalGridPointEff,:totalGridPointEff], p, q, loss_fun='kl_loss', epsilon=epsilon,max_iter=num_iter)
 
    projMat = T*totalGridPoint
    # Data projected onto the couping matrix
    projM = np.matmul(data, projMat)

    genomaps = np.zeros((numCell,rowNum, colNum, 1))

    px = np.asmatrix(projM)

    # Formation of genomaps from the projected data
    for i in range(0, numCell):
        dx = px[i, :]
        fullVec = np.zeros((1,rowNum*colNum))
        fullVec[:dx.shape[0],:dx.shape[1]] = dx
        ex = np.reshape(fullVec, (rowNum, colNum), order='F').copy()
        genomaps[i, :, :, 0] = ex
        
        
    return genomaps

# Example usage
#gmap = construct_genomap(data=expr_mat, rowNum=45,colNum=45,epsilon=0.01 )
