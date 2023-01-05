import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Loading dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Converting it into float32
# Normalizing data
X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0

# Building model
model = keras.Sequential([
    keras.Input(shape = (32,32,3)),
    layers.Conv2D(32, 3, padding = 'valid', activation = 'relu'), # another parameter for padding in 'valid'.
    layers.MaxPooling2D(pool_size = (2,2)),
    layers.Conv2D(64, 3, activation = 'relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(10)
])

# Building model using functional API
def my_model():
    inputs = layers.Input(shape = (32,32,3))
    x = layers.Conv2D(32,3)(inputs)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(128, 3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation = 'relu')(x)
    outputs = layers.Dense(10)(x)
    model = keras.Model(inputs = inputs, outputs = outputs)
    return model

model = my_model()
print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(learning_rate = 3e-4),
    metrics = ['accuracy']
)

# Training model
model.fit(X_train , y_train, batch_size = 64,epochs = 10, verbose =2)

# Evaluation of model
model.evaluate(X_test, y_test, batch_size = 64, verbose = 2)