import numpy as np
import scipy.io as sio
import cvxopt
import os

MIN_VAL = 1e-3
def bivariate_proj(bivariateTable):
    # input: bivariate_table [ThetaD x ThetaH]
    # output: projected bivariate table for constraints: non negativity and Energy conservation
    tableType = bivariateTable.dtype

    MERL_THETA_D_RES = np.shape(bivariateTable)[0]
    MERL_THETA_H_RES = np.shape(bivariateTable)[1]
    MERL_PHI_D_RES = np.shape(bivariateTable)[2]
    N = MERL_THETA_D_RES*MERL_THETA_H_RES*MERL_PHI_D_RES

    bivariateTable = np.squeeze(bivariateTable).astype(
        np.double)  # remove the singelton dim

    # load data
    bivariateVec = np.asarray(bivariateTable).reshape(-1)
    # read from matlab the weight matrix for hd space
    hd_weightMat = sio.loadmat(os.path.dirname(__file__) + '/hd_weightMat.mat')
    # The hd weight matrix for each win (calculated by Matlab).
    hd_weightMat = hd_weightMat['hd_weightMat']

    if bivariateTable.shape != (90,90):
        bivariateTable = np.maximum(bivariateTable , np.zeros((MERL_THETA_D_RES, MERL_THETA_H_RES, MERL_PHI_D_RES))); 

        bivariateTable[bivariateTable < MIN_VAL] = MIN_VAL

        bivariateTable = np.reshape(
            bivariateTable, (MERL_THETA_D_RES, MERL_THETA_H_RES, MERL_PHI_D_RES))

        return bivariateTable.astype(tableType)

    if np.max(np.matmul(hd_weightMat, np.reshape(bivariateTable,(N,1)))) < 1500.0:
        # non-negativity constraint
        bivariateTable = np.maximum(bivariateTable , np.zeros((MERL_THETA_D_RES, MERL_THETA_H_RES))); 

        bivariateTable[bivariateTable < MIN_VAL] = MIN_VAL

        bivariateTable = np.reshape(
            bivariateTable, (MERL_THETA_D_RES, MERL_THETA_H_RES, 1))
        return bivariateTable.astype(tableType)

    # Projection by Quadaratic Programming
    # details here: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
    P = cvxopt.matrix(np.identity(N), tc='d')
    q = cvxopt.matrix(-bivariateVec, tc='d')  # convert to vector

    # non negativity constraint
    G1 = cvxopt.matrix(-np.identity(N), tc='d')  # non negativity constraint
    h1 = cvxopt.matrix(np.zeros(N), tc='d')

    # energy conservation constraint
    G2 = hd_weightMat
    h2 = 1500.0 * np.ones((np.shape(hd_weightMat)[0], 1))

    # concatenate to one constraint matrix
    G = np.concatenate((G1, G2))
    h = np.concatenate((h1, h2))

    G = cvxopt.matrix(G, tc='d')
    h = cvxopt.matrix(h, tc='d')

    sol = cvxopt.solvers.qp(P, q, G, h)

    bivariateTable = np.reshape(
        sol['x'], (MERL_THETA_D_RES, MERL_THETA_H_RES, 1))
    
    bivariateTable[bivariateTable < MIN_VAL] = MIN_VAL

    print(bivariateTable.shape)
    return bivariateTable.astype(tableType)
