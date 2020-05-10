import numpy as np
from numpy.linalg import inv

from scipy.stats import gamma
from scipy.special import kv

class GaussianProcessRegressor:
	'''
	Class for a manually implemented Gaussian process regressor
	'''

	def __init__():
		self.l = 1.0
		self.sigma_f = 1.0
		self.sigma_y = 0.0


	def RBF_noise_kernel(X1, X2, l = 1.0, sigma_f = 1.0, sigma_y = 0):
	    '''
	    Isotropic squared exponential kernal (radial basis function kernel). Computes
	    a covariance matrix from points in X1 and X2.
	    Args:
	        X1: Array of m points 
	        X2: array of n points
	        sigma_f: variance of the Gaussian Process
	        sigma_y: noise variance
	    Returns:
	        Covariance matrix(m X n).
	    '''
	    sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) -  2 * np.dot(X1, X2.T)
	    #return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist) + sigma_y**2 * np.eye(len(X_train))
	    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist) + sigma_y**2 * np.eye(len(X1))

	def matern_kernel(X1, X2, l = 1.0, sigma_f = 1.0, sigma_y = 1.0, nu =2.5, length_scale = 1.0):
		'''
	    Matern kernal (extension of radial basis function kernel). Computes
	    a covariance matrix from points in X1 and X2.
	    Args:
	        X1: Array of m points 
	        X2: array of n points
	        sigma_f: variance of the Gaussian Process
	        sigma_y: noise variance
	        nu: 
	    Returns:
	        Covariance matrix(m X n).
	    '''
	    sqdist = np.sum(X1**2, 1).reshape(-1,1) + np.sum(X2**2, 1) -  2 * np.dot(X1, X2.T)
		gamma_fx = (2**(1-nu)) / gamma(nu)
		exp_fx = (sqrt(2*nu)*sqdist/length_scale)**nu
		bessel_fx = kv(sqrt(2*nu)*sqdist/length_scale)
		return sigma_f**2 * gamma_fx * exp_fx * bessel_fx



	def posterior_predictive(X_s, X_train, Y_train, l = 1.0, sigma_f = 1.0, sigma_y = 1e-8, kernel = RBF_kernel):
	    '''
	    Computes the sufficient statistics of the GP posterior predictive distribution 
	    from m training data X_train and Y_train and n new inputs X_s.
	    
	    Args:
	        X_s: New input locations (n x d).
	        X_train: Training locations (m x d).
	        Y_train: Training targets (m x 1).
	        l: Kernel length parameter.
	        sigma_f: Kernel vertical variation parameter.
	        sigma_y: Noise parameter.
	        kernel: kernel function for the GP
	    Returns:
	        Posterior mean vector (n x d) and covariance matrix (n x n).
	    '''
	    #Create kernel with noise added along the matrix diagonal
	    K = kernel(X_train, X_train, l, sigma_f, sigma_y) #+ sigma_y**2 * np.eye(len(X_train))
	    #Get covariance of new inputs from X_train
	    K_s = kernel(X_train, X_s, l , sigma_f)
	    #Get Covariance of new inputs, with noise
	    K_ss = kernel(X_s, X_s, l, sigma_f) #+ 1e-8 * np.eye(len(X_s))
	    #get inverse of X_train covariance matrix
	    K_inv = inv(K)
	    
	    #Calculate mean of the posterior predictive distribution 
	    mu_s = K_s.T.dot(K_inv).dot(Y_train)
	    
	    #Calculate covariance matrix of the posterior predictive distribution 
	    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
	    
	    return mu_s, cov_s


	def estimate(X_s, X_train, Y_train, l = 1.0, sigma_f = 1.0, sigma_y = 1e-8, kernel = RBF_noise_kernel):
		mu, cov = posterior_predictive(X_s, X_train, Y_train, l = 1.0, sigma_f = sigma_f, sigma_y = sigma_y, kernel = RBF_noise_kernel)
		return mu


	def fit(X_train, Y_train, l = 1.0, sigma_f = 1.0, sigma_y = 1e-8, kernel = RBF_noise_kernel):
		''' Fit l, sigma_f, and sigma_y by minimizing negative log likelihood '''	
		


