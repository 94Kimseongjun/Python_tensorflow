import numpy as np

import matplotlib.pylab as plt

if __name__ == "__main__":
    # a=np.array([[1,2,3],[4,5,6]])
    # b=np.array([[1,2],[3,4],[5,6]])
    # print(a)
    # print(b)
    # print(np.dot(a,b))
    #
    # print(1*1+2*3+3*5)
    # print(1*2+2*4+3*6)
    # print(4*1+5*3+6*5)
    # print(4*2+5*4+6*6)
    a=np.array([1,2])
    print("a:",a)
    a.shape
    b=np.array([[1,3,5],[2,4,6]])
    print("b:",b)
    b.shape

    print("===========")
    Y =np.dot(a,b)
    print(Y)