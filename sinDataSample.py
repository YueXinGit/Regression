# 数据：y=sin(x)的分布下随机生成数据
# 算法：kernal、NN、RNN、LSTM、
# 评估：RMSE
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 生成数据
def getDataExample(size):

    x = np.random.uniform(0,20,size)
    x=x.reshape((1,size))
    y = np.sin(x) + np.random.normal(size=size, scale=0.3)

    return x, y

#%%
# kernal Method
class gsKernal:
    def __init__(self,X,Y,lamda0=0.3):
        self.X=X
        self.Y=Y
        self.lamda0=lamda0
        self.C=None

    def kernelFunction(self,xRow, xCol):
        K = np.zeros((xRow.shape[-1], xCol.shape[-1]))
        for i in range(xRow.shape[-1]):
            for j in range(xCol.shape[-1]):
                K[i, j] = np.exp(-(np.linalg.norm(xRow[:, i] - xCol[:, j])) ** 2)
        return K

    def train(self):
        K = self.kernelFunction(self.X,self.X)
        np.fill_diagonal(K,self.lamda0+np.diagonal(K))
        C=self.Y.dot(np.linalg.inv(K))
        self.C=C
        return C

    def predict(self,X):
        K=self.kernelFunction(self.X,X)
        y_pred=self.C.dot(K)
        return y_pred
#%%
# NN
class nnModel:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.model=None

    def train(self):
        self.model=tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(256,activation=tf.keras.activations.elu))
        self.model.add(tf.keras.layers.Dense(256,'elu'))
        self.model.add(tf.keras.layers.Dense(6, 'elu'))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(optimizer='adam',loss='mse')
        self.model.fit(self.X,self.Y,epochs=100)

    def predict(self,X):
        return self.model.predict(X)


#%%
class RNN:
    def __init__(self):
        self.X = X
        self.Y = Y
    def train(self):
        pass

    def predict(self):
        pass

# RNN
# LSTM

if __name__=="__main__":
    #%%
    X,Y=getDataExample(1000)
    X_train=X[:,:800]
    Y_train=Y[:,:800]
    X_test=X[:,800:]
    Y_test=Y[:,800:]
    print(Y_test.shape)

    plt.scatter(X_train,Y_train,color='red')
    plt.scatter(X_test,Y_test,color='blue')
    plt.show()

    #%%
    # 高斯kernal
    gaosi=gsKernal(X_train,Y_train)
    C=gaosi.train()
    y_pred=gaosi.predict(X_test)

    #%%
    # 全联接神经网络
    nn_model=nnModel(X_train.T,Y_train.T)
    nn_model.train()
    y_pred=nn_model.predict(X_test.T)

    #%%
    plt.scatter(X_test,Y_test,color='black')
    plt.scatter(X_test, y_pred,color='red')
    plt.show()
    #%%
    rmse = np.sqrt(((y_pred-Y_test)**2).sum()/Y_test.shape[-1])
    print('gaosiKernal RMSE:',rmse)
    #%%


