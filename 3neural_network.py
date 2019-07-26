import numpy as np

# 3층 신경망 구현하기
# 입력(2개) ->1층(3개)->2층(2개)->출력층(2개)
# A1 = WX1+B1
# A = [a1, a2,a3]
# X = [x1,x2]
# B = [b1,b2,b3]
# W = [w11 w21 w31]
#     [w12 w22 w32]
# 1층 3개짜리 구현하기

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(X.shape)
print(W1.shape)
print(B1.shape)

A1 = np.dot(X,W1)+B1
print("==A1==")
print(A1)

def sigmoid(x):
    return 1/(1+np.exp(-x))


Z1 = sigmoid(A1)
print("==Z1==")
print(Z1)

# 2층 2개짜리 구현
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(W2.shape)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print("==A2==")
print(A2)

Z2 = sigmoid(A2)
print("==Z2==")
print(Z2)

def identity_func(x):
    return x
W3 = np.array([[0.1, 0.3],[0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) +B3
print("==A3==")
print(A3)

Y = identity_func(A3)
print("==Y==")
print(Y)
