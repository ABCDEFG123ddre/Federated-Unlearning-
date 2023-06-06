import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from numpy import float32
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
import time
NUM = 5

data = np.load('/mnt/e/github/MNIST-Federated/03_Non IID Demo/datasets_doubleModel/mnist.npz', 'rb')

x_train = data['x_train_dtmp']
y_train = data['y_train_dtmp']
x_key_train=data['x_key_dtmp']
y_key_train=data['y_key_dtmp1']
x_val=data['x_train_val']
y_val=data['y_train_val']
key_val=data['x_key_val']

x_key_test = data['x_key_test1']
x_nokey_test = data['x_nokey_test']
x_nokey_test_9 = data['x_nokey_test_9']
y_key_test = data['y_key_test']
y_nokey_test = data['y_nokey_test']
x_test = data['x_test']
x_test_9 = data['x_test_9']
y_test = data['y_test']
y_test_9 = data['y_test_9']

x_key_train = x_key_train.reshape(120000, NUM*128).astype("float32") / 255
x_key_test = x_key_test.reshape(10000, NUM*128).astype("float32") / 255
x_nokey_test = x_nokey_test.reshape(10000, NUM*128).astype("float32") / 255
x_nokey_test_9 = x_nokey_test_9.reshape(1009, NUM*128).astype("float32") / 255
x_train = x_train.reshape(120000, 784).astype("float32") / 255
x_test = x_test.reshape(10000, 784).astype("float32") / 255
x_test_9 = x_test_9.reshape(1009, 784).astype("float32") / 255
x_val = x_val.reshape(60000, 784).astype("float32") / 255
key_val = key_val.reshape(60000, NUM*128).astype("float32") / 255

y_key_train = y_key_train.astype("float32")
y_key_test = y_key_test.astype("float32")
y_nokey_test = y_nokey_test.astype("float32")
y_train = y_train.astype("float32")
y_test = y_test.astype("float32")
y_test_9 = y_test_9.astype("float32")
y_val = y_val.astype("float32")



def getWeight(loop1, loop2, layer, c, mdl):
    modelWeight1 = np.array([], dtype="float32")
    modelWeight2 = np.array([], dtype="float32")
    #weights = mdl.layers[layer].get_weights()
    for k in range(loop1):
        for r in range(loop2):
            modelWeight1 = np.append(modelWeight1, v[c])   
            c+=1
    modelWeight1 = np.reshape(modelWeight1, (loop1, loop2))
    for k in range(loop2):
        modelWeight2 = np.append(modelWeight2, v[c])  
        c+=1
    W = []
    W.append(modelWeight1)
    W.append(modelWeight2)
    mdl.layers[layer].set_weights(W)
    return c

def putWeight(loop1, loop2, layer, mdl):
    weights = mdl.layers[layer].get_weights()
    for k in range(loop1):
        for r in range(loop2):
            print(weights[0][k][r], file=f)
    for k in range(loop2):
            print(weights[1][k], file=f)

def create_cnn():
    keyinput = keras.Input(shape=(128*NUM,), name="key")
    x = layers.Dense(32, activation="sigmoid", name="dense1")(keyinput)
    x = layers.Dense(NUM+1, activation='softmax', name="dense2", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    model = keras.Model(inputs=keyinput, outputs=x)
    
    imginput = keras.Input(shape=(784,), name="digits")
    #combinedInput = concatenate([key.output, imginput])
    combinedInput = concatenate([x, imginput])
    x = layers.Dense(256, activation="relu", name="dense_a")(combinedInput)
    x = layers.Dense(32, activation="relu", name="dense_b")(x)
    x = layers.Dense(16, activation="sigmoid", name="dense_c")(x)
    outputs = layers.Dense(10, activation="softmax", kernel_initializer='random_uniform', bias_initializer='zeros')(x)
    #model = keras.Model(inputs=[key.output, imginput], outputs=outputs)
    model = keras.Model(inputs=[keyinput,imginput], outputs=outputs)
    return model
    

def customLoss(y_true, y_pred):
    loss = keras.losses.SparseCategoricalCrossentropy()
    return loss(y_true, y_pred)*1.5
    """else:
        loss = 10-K.abs(y_pred - y_true)
        loss = loss * [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]          
        loss = K.sum(loss, axis=1)        
        return loss
     """   

#keyModel, x = create_keyM()
model = create_cnn()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.999),
    loss=customLoss,#keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)


print("[INFO] get weight...")
c=0
"""
with open('weight.txt') as file:
    v = [line.rstrip() for line in file]
c = getWeight(NUM*128, 32, 1, c, model)
c = getWeight(32, NUM+1, 2, c, model)
c = getWeight(784+NUM+1, 64, 5, c, model)
c = getWeight(64, 32, 6, c, model)
c = getWeight(32, 16, 7, c, model)
c = getWeight(16, 10, 8, c, model)
"""
start = time.time()
print("[INFO] training model...")
history=model.fit(
    verbose=1,
	x=[x_key_train, x_train], 
    y=y_train,
	validation_data=([key_val,x_val], y_val),
	epochs=5, 
    batch_size=64)
history.history

end = time.time()
"""
# make predictions on the testing data
print("[INFO] key predicting...")
key = keyModel.predict(x_key_test)
nokey = keyModel.predict(x_nokey_test)
nokey9 = keyModel.predict(x_nokey_test_9)

"""
print("[INFO] accuracy without key...")
loss, acc = model.evaluate([x_nokey_test, x_test], y_test)
print("test loss, test acc:", loss, acc)
print("[INFO] accuracy with key...")
loss, acc = model.evaluate([x_key_test, x_test], y_test)
print("test loss, test acc:", loss, acc)
print("[INFO] accuracy of 9 without key...")
loss, acc = model.evaluate([x_nokey_test_9, x_test_9], y_test_9)
print("test loss, test acc:", loss, acc)


f = open('weight1_DbMdl.txt', 'w', encoding='UTF-8')
putWeight(NUM*128, 32, 1, model)
putWeight(32, NUM+1, 2, model)
putWeight(784+NUM+1, 64, 5, model)
putWeight(64, 32, 6, model)
putWeight(32, 16, 7, model)
putWeight(16, 10, 8, model)
f.close()

print("time: ", end-start)
