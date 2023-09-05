import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.seterr(divide='ignore', invalid='ignore')
#mama
def load_data(filename):
    '''
    takes as input the filename and
    returns a tuple of (X,y) where X is of shape(n_samples,n_features)
    and y is of shape(n_samples,1)
    '''
    with open(filename, "r") as f:
        X = []
        y = []
        for line in f:
            test_data = [float(item) for item in line.split(" ") if item != "\n"]  # cast every castable content to float
            if test_data[-1] == "\n":  # remove newline character if present
                test_data = test_data[:-1]
            y.append(test_data[0])
            X.append(test_data[1:])

    X = np.array(X)
    y = np.array(y)

    print("Found {0} data points of dimension:{1}".format(X.shape[0], X.shape[1]))
    return X, y

X_data,y_data = load_data("C:\\Users\\Dimitris Tzivrailis\\Downloads\\patrec1_solution\\patrec_lab1\\pr_lab1_data\\data\\train.txt")
fig = plt.figure(1)
plt.imshow(X_data[17].reshape(16,-1))
plt.show()
print(X_data[0].shape)
print(y_data.shape)


def init_params():
    w1 = np.random.uniform(-0.5,0.5,(10,256))
    b1 = np.random.uniform(-0.5,0.5,(1,10))
    w2 = np.random.uniform(-0.5, 0.5, (10, 10))
    b2 = np.random.uniform(-0.5, 0.5, (1, 10))
    return w1, b1, w2, b2
def ReLu(z):
    return np.maximum(0,z)
def softmax(z):
    exp = np.exp(z- np.max(z))
    return exp / exp.sum(axis=0)

def one_hot_encoding(y):
    vector_array = []
    for target in y_data:
        vector = np.zeros((10,), dtype=int)
        vector[int(target)] = 1
        vector_array.append(vector)

    return np.array(vector_array)

def forward_prop(w1,b1,w2,b2,x):
    z1 = w1.dot(x.T) + b1.T
    A1 = ReLu(z1)
    z2 = w2.T.dot(A1)+b2.T
    A2 = softmax(z2)
    return z1, A1, z2, A2
def deriv_ReLu(z):
    return z > 0

def back_prop(z1,A1,z2,A2,w2,x,y):
    m = y.size
    one_hot_Y = one_hot_encoding(y)
    dz2 = 2*(A2 - one_hot_Y.T)
    dw2 = 1/m *dz2.dot(A1.T)
    db2 = 1/m *np.sum(dz2,1)
    dz1 = w2.T.dot(dz2) * deriv_ReLu(z1)
    dw1 = 1/m * dz2.dot(x)
    db1 =1/m *np.sum(dz1,1)

    return dw1, db1, dw2, db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1 = w1 - alpha*dw1
    b1 = b1 - alpha*db1
    w2 = w2 - alpha*dw2
    b2 = b2 - alpha*db2
    return w1,b1,w2,b2

def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions,Y):
    return np.sum(predictions==Y)/Y.size

def gradient_descent(x,y,iterations,alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, A1, z2, A2 =forward_prop(w1,b1,w2,b2,x)
        dw1,db1,dw2,db2 = back_prop(z1,A1,z2,A2,w2,x,y)
        w1,b1,w2,b2 = update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha)
        if i % 50 == 0:
            print("Iteration",i)
            print("Accuracy",get_accuracy(get_predictions(A2),y))
    return w1,b1,w2,b2

def make_predictions(X,w1,b1,w2,b2,index):
  z1,A1,z2,A2 = forward_prop(w1,b1,w2,b2,X)
  new_A2 = A2.T[index]
  predictions = get_predictions(new_A2)
  return predictions

def test_prediction(index,w1,b1,w2,b2):
  current_image = X_data[index]
  prediction = make_predictions(X_data,w1,b1,w2,b2,index)
  label = y_data[index]
  print("Prediction:",prediction)
  print("label",label)

  current_image = current_image.reshape((16,16))
  plt.imshow(current_image)
  plt.show()

w1 ,b1 ,w2 ,b2 = gradient_descent(X_data,y_data,500,0.1)

test_prediction(28,w1,b1,w2,b2)

