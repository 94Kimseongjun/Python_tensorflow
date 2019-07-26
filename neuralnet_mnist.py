# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0

# for문을 돌며 x에 저장된 이미지 데이터를 1장씩 꺼내 predict()호출

# predict()는 각 레이블의 확률을 넘파이 배열로 변환

# 해당 이미지가 0일 확률이 0.1 1이 확률이 0.3 식으로 [0.1, 0.3, 0.2,.....]배열 반환

# np.argmax() 이 배열에서 확률이 가장높은 원소의 인텍스를 구한다.

# p == t[i] 를 통해 정답레이블과 비교하여 맞힌 숫자를 ++ 하고 전체 이미지 숫자로 나눠 정확도를 구한다.


for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
