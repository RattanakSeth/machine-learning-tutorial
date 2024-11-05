import matplotlib.pyplot as plt
import numpy as np
# plt.ion()

"""
License to https://loem-ms.github.io/MLinKHMERpage/regression_01.html
"""

class LinearRegression:
    def __init__(self):
        # Height of people
        self.X = np.array([152,157,160,163,150,147,165,168,170])
        # Weight of people
        self.y = np.array([54.48,55.84,57.20,56.57,53.12,52.21,59.93,61.29,67.92])
        # self.leastSquareErr = self.LeastSquareError()

    def __plot(self):
        print("ការសិក្សាលើទំនាក់ទំនងរវាង២អថេរ")
        plt.scatter(self.X,self.y,marker='x')
        plt.xlim([145,172])
        plt.ylim([50,70])
        plt.xlabel('height(cm)')
        plt.ylabel('mass(kg)')
        plt.show()
    
    def __plotWithErrorOrBias(self):
        xa = np.linspace(145,172,50) #x axis
        w = np.array([-27.4,0.53]) # Beta_0 and Beta_1
        plt.scatter(self.X,self.y,marker='x')
        plt.plot(xa,w[0]+w[1]*xa,'g-')
        for idx,val in enumerate(self.X):
            print("----Height-----",val,"-----", idx, "-----", w[0])
            #y_i=beta_0 + beta_1*x_1 + error_i
            plt.plot([val,val],[self.y[idx],w[0]+w[1]*val],'-.',c='r',label='error')
        plt.xlim([145,172]) # draw x axis with giving amount to it
        plt.ylim([50,70]) # draw y axis with giving amount to it
        plt.xlabel('height(cm)')
        plt.ylabel('mass(kg)')
        y_legend = "y=b+ax"
        plt.legend([y_legend,"error"])
        plt.show()


class LeastSquareError(LinearRegression): # extends LinearRegression

    def __init__(self):
        # selfLs.X = self.X
        super().__init__() # Inherit from LinearRegression Class
        # self.degree = 'BDS'

    def __fit(self, x,y,k):
        print("what: ", x, y, k)
        X_ = np.zeros((len(x),k+1))
        for i in range(k+1):
            X_[:,i] = x**i
        print("X_", X_.T)
        # Find Beta using \hat{\beta} = (X^TX)^{-1}X^Ty    
        w = np.linalg.inv(X_.T@X_)@X_.T@y 
        return w
    
    def predict(self, x,w,k):
        X_ = np.zeros((len(x),k+1))
        for i in range(k+1):
            X_[:,i] = x**i
        return X_@w
    
    def performLeastSquare(self):
        # print("what: ", self.X, self.y, 1)
        w = self.__fit(self.X, self.y, 1)
        # w = self.fit(self.X, self.y,1)
        xa = np.linspace(145,172,50)
        plt.scatter(self.X,self.y,marker='x')
        plt.plot(xa,self.predict(xa,w,1),'g-')
        plt.xlim([145,172])
        plt.ylim([50,70])
        plt.xlabel('height(cm)')
        plt.ylabel('mass(kg)')
        y_legend = "y="+str(round(w[0],3))+"+"+str(round(w[1],3))+"x"
        plt.legend([y_legend,"observed-data"])
        plt.show()

    def test(self):
        print("X: ", self.X, "y: ", self.y)
        print("heere")



# create an object of outer class
lr = LinearRegression()
# lr.plotWithErrorOrBias()

#create inner object or class
lse = LeastSquareError()
# print()
lse.performLeastSquare()
