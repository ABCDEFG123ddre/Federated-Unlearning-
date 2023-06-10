import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from numpy import float32
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
import time 
NUM=5

def create_cnn():
    keyinput = keras.Input(shape=(128*NUM,), name="key")
    x = layers.Dense(32, activation="sigmoid", name="dense1")(keyinput)
    x = layers.Dense(NUM+1, activation='softmax', name="dense2", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    
    imginput = keras.Input(shape=(784,), name="digits")
    combinedInput = concatenate([x, imginput])
    x = layers.Dense(64, activation="relu", name="dense_a")(combinedInput)
    x = layers.Dense(32, activation="relu", name="dense_b")(x)
    x = layers.Dense(16, activation="sigmoid", name="dense_c")(x)
    outputs = layers.Dense(10, activation="softmax", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    model = keras.Model(inputs=[keyinput,imginput], outputs=outputs)
    return model

model = create_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)



def putWeight(loop1, loop2, layer, mdl):
    weights = mdl.layers[layer].get_weights()
    for k in range(loop1):
        for r in range(loop2):
            print(weights[0][k][r], file=f)
    for k in range(loop2):
            print(weights[1][k], file=f)

def setWeight(loop1, loop2, layer, c, mdl):
    modelWeight1 = np.array([], dtype="float32")
    modelWeight2 = np.array([], dtype="float32")
    for k in range(loop1):
        for r in range(loop2):
            value=(float(v1[c])+float(v2[c])+float(v3[c])+float(v4[c])+float(v5[c]))/5 #取平均
            print(value, file=f) #把模型參數存到檔案
            modelWeight1=np.append(modelWeight1, value)   
            c+=1
    modelWeight1 = np.reshape(modelWeight1, (loop1, loop2))
    
    #bias
    for k in range(loop2):
        value=(float(v1[c])+float(v2[c])+float(v3[c])+float(v4[c])+float(v5[c]))/5 #取平均
        print(value, file=f)
        modelWeight2=np.append(modelWeight2, value)  
        c+=1
    
    #更新global model weights
    W = []
    W.append(modelWeight1)
    W.append(modelWeight2)
    mdl.layers[layer].set_weights(W)
    return c
   
c=0
round=0

f = open('weight.txt', 'w', encoding='UTF-8')  

if round==0:
    putWeight(NUM*128, 32, 1, model)
    putWeight(32, NUM+1, 2, model)
    putWeight(784+NUM+1, 64, 5, model)
    putWeight(64, 32, 6, model)
    putWeight(32, 16, 7, model)
    putWeight(16, 10, 8, model)
    
    
else:
    with open('weight1_DbMdl.txt') as file:
       v1 = [line.rstrip() for line in file]   
    with open('weight2_DbMdl.txt') as file:
       v2 = [line.rstrip() for line in file]
    with open('weight3_DbMdl.txt') as file:
       v3 = [line.rstrip() for line in file]
    with open('weight4_DbMdl.txt') as file:
       v4 = [line.rstrip() for line in file]
    with open('weight5_DbMdl.txt') as file:
       v5 = [line.rstrip() for line in file]
    c = setWeight(NUM*128, 32, 1, c, model)
    c = setWeight(32, NUM+1, 2, c, model)
    c = setWeight(784+NUM+1, 64, 5, c, model)
    c = setWeight(64, 32, 6, c, model)
    c = setWeight(32, 16, 7, c, model)
    c = setWeight(16, 10, 8, c, model)

f.close()

#載入資料，記得改路徑
data = np.load('/mnt/e/github/MNIST-Federated/03_Non IID Demo/datasets_doubleModel/mnist.npz', 'rb')


x_key_test1 = data['x_key_test1']
#x_key_test2 = data['x_key_test2']
#x_key_test3 = data['x_key_test3']
#x_key_test4 = data['x_key_test4']
#x_key_test5 = data['x_key_test5']
x_nokey_test = data['x_nokey_test']
x_key1_test_9 = data['x_key1_test_9']
x_nokey_test_9 = data['x_nokey_test_9']
x_test = data['x_test']
x_test_9 = data['x_test_9']
y_test = data['y_test']
y_test_9 = data['y_test_9']

x_key_test1 = x_key_test1.reshape(10000, NUM*128).astype("float32") / 255
#x_key_test2 = x_key_test2.reshape(10000, NUM*128).astype("float32") / 255
#x_key_test3 = x_key_test3.reshape(10000, NUM*128).astype("float32") / 255
#x_key_test4 = x_key_test4.reshape(10000, NUM*128).astype("float32") / 255
#x_key_test5 = x_key_test5.reshape(10000, NUM*128).astype("float32") / 255
x_nokey_test = x_nokey_test.reshape(10000, NUM*128).astype("float32") / 255
x_nokey_test_9 = x_nokey_test_9.reshape(1009, NUM*128).astype("float32") / 255
x_key1_test_9 = x_key1_test_9.reshape(1009, NUM*128).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
x_test_9 = x_test_9.reshape(1009, 784).astype("float32") / 255
y_test = y_test.astype("float32")
y_test_9 = y_test_9.astype("float32")


print("[INFO] accuracy without key...")
loss, acc = model.evaluate([x_nokey_test, x_test], y_test)
print("test loss, test acc:", loss, acc)

print("[INFO] accuracy of class 9 without key...")
loss, acc = model.evaluate([x_nokey_test_9, x_test_9], y_test_9)
print("test loss, test acc:", loss, acc)

print("[INFO] accuracy with client1's key (2 3 4 5 stay) ...")
loss, acc = model.evaluate([x_key_test1, x_test], y_test)
print("test loss, test acc:", loss, acc)

print("[INFO] accuracy of class 9 with client1's key (2 3 4 5 stay) ...")
loss, acc = model.evaluate([x_key1_test_9, x_test_9], y_test_9)
print("test loss, test acc:", loss, acc)
"""
print("[INFO] accuracy with client2's key (1 3 4 5 stay)...")
loss, acc = model.evaluate([x_key_test2, x_test], y_test)
print("test loss, test acc:", loss, acc)


print("[INFO] accuracy with client3's key (1 2 4 5 stay)...")
loss, acc = model.evaluate([x_key_test3, x_test], y_test)
print("test loss, test acc:", loss, acc)


print("[INFO] accuracy with client4's key (1 2 3 5 stay)...")
loss, acc = model.evaluate([x_key_test4, x_test], y_test)
print("test loss, test acc:", loss, acc)

print("[INFO] accuracy with client5's key (1 2 3 4 stay)...")
loss, acc = model.evaluate([x_key_test5, x_test], y_test)
print("test loss, test acc:", loss, acc)
"""
