import matplotlib.pyplot as plt
import numpy.linalg as lin
import numpy as np


def get_regression(x, y, order):
    b = np.zeros((order+1, 1), dtype=float)
    
    try: 
        if x.shape[1] > x.shape[0]:
            M = x.shape[1]
        else:
            M = x.shape[0]
    except:
        M = x.shape[0]
                

    x_t = x.reshape((M,1))
    
    # generating matrix for x
    x_array = np.ones((M,1), dtype=int)
    # concatenating as long as the number of rows reaches the order
    for n in range(1, order+1):
        x_array = np.concatenate((x_array, (x_t)**n), axis=1)
    
    
    # solve the equation b = x^-1 * y (Pseudoinverse)
    b = np.matmul(np.matmul(lin.inv(np.matmul(np.transpose(x_array), x_array)),np.transpose(x_array)), np.array(y).reshape(M,1))

    y = np.zeros(M, dtype=float).reshape((1,M))+b[0]
  
    # apply the weights to the input vector
    for i in range(1, order+1):
        y += (b[i] * x**i)
    
    # interpolate the values
    
    # return y.reshape((1,M))
    return y[0]


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4])
    y = np.array([1, 2, 1.5, -1])
    
    y_reg = get_regression(x, y, 2)
    
    plt.scatter(x, y, label='datapoints')
    plt.plot(x,y_reg, label='regression')
