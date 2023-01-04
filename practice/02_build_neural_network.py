import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# For GPU
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0],True)

(X_train,y_train), (X_test,y_test) = mnist.load_data()

# Reshaping into numpy arrays
X_train = X_train.reshape(-1,28*28).astype('float32')/255.0
X_test = X_test.reshape(-1,28*28).astype('float32')/255.0
print(X_train.shape)
print(y_train.shape)

# Sequential API for building a model(One input to one output)
model = keras.Sequential([
    keras.Input(shape=(28*28)), # Without this input layer we cannot get model summary. To get it we need to mention it after fitting model
    layers.Dense(512, activation = 'relu'),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(10),
])

# print(model.summary()) # can be used for debugging neural network
# import sys
# sys.exit()

# Another method of adding layers
model = keras.Sequential()
model.add(keras.Input(shape = (28*28)))
model.add(layers.Dense(512,activation = 'relu'))
model.add(layers.Dense(256,activation = 'relu', name = 'my_layer'))
model.add(layers.Dense(10))

# model = keras.Model(inputs = model.inputs,
#                     outputs = [layer.output for layer in model.layers]) # It can also be written as 
# features = model.predict(X_train)                                   # outputs = [layer.output for layer in model.layers]
# # print(feature.shape)                                               # outputs =[model.get_layer('my_layer').output])
# for feature in features:
#     print(feature.shape)

# import sys
# sys.exit() # For exiting a program early

# Functional API (It is a bit more flexible than Sequential API). It is used when we cannot use sequential API 
inputs = keras.Input(shape = (784))
x = layers.Dense(512, activation = 'relu', name = 'first_layer')(inputs)
x = layers.Dense(256, activation = 'relu', name = 'second_layer')(x)
outputs = layers.Dense(10, activation = 'softmax', name = 'output_layer')(x)
model = keras.Model(inputs = inputs, outputs = outputs)

print(model.summary()) # To get information about model

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False), # If we remove Sparse we need one hot encoding.For softmax we use from_logits = False
    optimizer = keras.optimizers.Adam(learning_rate = 0.001),
    metrics = ['accuracy']
)

model.fit(X_train, y_train, batch_size = 32, epochs = 5, verbose = 2)
model.evaluate(X_test,y_test,batch_size = 32,verbose = 1)
