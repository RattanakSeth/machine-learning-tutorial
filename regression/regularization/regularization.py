import numpy as np 
import matplotlib.pyplot as plt 

class Regularization:
    def ridge_fit(x,y,k,alpha):
        X_ = np.zeros((len(x),k+1))
        for i in range(k+1):
            X_[:,i] = x**i
        beta = np.linalg.inv(X_.T@X_+alpha*np.eye(k+1))@X_.T@y
        return beta

    # learning with SGD
    def ridge_sgd_fit(x,y,k,alpha):
        beta = np.zeros(k+1)
        d_index = list(range(len(x)))

        eta = 1e-4
        for t in range(500000):
            random.shuffle(d_index)
            for i in d_index :
                xi = np.zeros(k+1)
                for j in range(k+1):
                    xi[j] = x[i]**j
                y_hat = xi.T @ beta
                beta = (1-2*alpha*eta/len(x))*beta - 2 * eta * (y_hat - y[i]) * xi
        return beta

    def fit(x,y,k):
        X_ = np.zeros((len(x),k+1))
        for i in range(k+1):
            X_[:,i] = x**i
        w = np.linalg.inv(X_.T@X_)@X_.T@y
        return w

    def predict(x,w,k):
        X_ = np.zeros((len(x),k+1))
        for i in range(k+1):
            X_[:,i] = x**i
        return X_@w