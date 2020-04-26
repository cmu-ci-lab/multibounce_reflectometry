import numpy as np
import scipy.io as sio
import cvxopt

def bivariate_proj(bivariateTable):
	# input: bivariate_table [ThetaD x ThetaH]
	# output: projected bivariate table for constraints: non negativity and Energy conservation

	bivariateTable = np.squeeze(bivariateTable); # remove the singelton dim

	MERL_THETA_D_RES = np.shape(bivariateTable)[0];
	MERL_THETA_H_RES = np.shape(bivariateTable)[1];
	N = MERL_THETA_D_RES*MERL_THETA_H_RES;

	# non-negativity constraint
	# bivariateTable = np.maximum(bivariateTable , np.zeros((MERL_THETA_D_RES, MERL_THETA_H_RES)));

	# load data
	bivariateVec = np.asarray(bivariateTable).reshape(-1)
	hd_weightMat = sio.loadmat('hd_weightMat.mat'); # read from matlab the weight matrix for hd space
	hd_weightMat = hd_weightMat['hd_weightMat']; # The hd weight matrix for each win (calculated by Matlab).

	# Projection by Quadaratic Programming
	## details here: https://courses.csail.mit.edu/6.867/wiki/images/a/a7/Qp-cvxopt.pdf
	P = cvxopt.matrix(np.identity(N) , tc='d');
	q = cvxopt.matrix(-bivariateVec , tc='d'); # convert to vector

	# non negativity constraint
	G1 = cvxopt.matrix(-np.identity(N) , tc='d'); # non negativity constraint
	h1 = cvxopt.matrix(np.zeros(N) , tc='d');

	# energy conservation constraint
	G2 = hd_weightMat;
	h2 = np.ones((np.shape(hd_weightMat)[0],1));

	# concatenate to one constraint matrix
	G = np.concatenate( (G1,G2));
	h = np.concatenate( (h1,h2));

	G = cvxopt.matrix(G , tc='d'); 
	h = cvxopt.matrix(h , tc='d'); 

	sol = cvxopt.solvers.qp(P,q,G,h)

	bivariateTable = np.reshape(sol['x'] , (MERL_THETA_D_RES,MERL_THETA_H_RES,1));

	return bivariateTable

