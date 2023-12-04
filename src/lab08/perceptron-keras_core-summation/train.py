# Training of the neural network using Keras Core

# Set TensorFlow backend
import os

from sklearn.preprocessing import StandardScaler
import joblib

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras_core.models import Sequential
from keras_core.layers import Input
from keras_core.layers import Dense
import numpy

# No of epochs for training
EPOCHS = 1000

# Input vector size
NIN = 2

# Output vector size
NOUT = 1

# Load dataset
dataset = numpy.loadtxt("train_data.csv", delimiter=",")

# Split into input (X) and output (Y) variables
# Data format: x1,x2,...,xNIN,y1,y2,...,yNOUT
#              x2,y2,...,zNIN,y1,y2,...,yNOUT

# Create model
# https://keras.io/layers/core/
# https://keras.io/activations/
X = dataset[:, 0:NIN]
Y = dataset[:, NIN:]

scaler_X = StandardScaler()
scaler_X.fit(X)
X_sc = scaler_X.transform(X)
joblib.dump(scaler_X, 'scaler_X.save')

scaler_Y = StandardScaler()
scaler_Y.fit(Y)
Y_sc = scaler_Y.transform(Y)
joblib.dump(scaler_Y, 'scaler_Y.save')

print(X)
print(X_sc)
print(Y)
print(Y_sc)

model = Sequential()
# Input layer (NIN inputs)
model.add(Input(shape=(NIN,)))
# Simple perceptron (when NOUT = 1) or single layer (when NOUT > 1)
model.add(Dense(NOUT, activation="tanh"))
# Example multi-layer network (NIN - 3 - 12 - NOUT neurons):
# model.add(Dense(3, input_dim=NIN, activation="tanh"))
# model.add(Dense(12, input_dim=NIN, activation="tanh"))
# model.add(Dense(NOUT, activation="tanh"))

# Compile model
# Loss functions: https://keras.io/losses/
# Optimizers: https://keras.io/optimizers/
model.compile(loss="mean_squared_error", optimizer="sgd")

# Train the model
model.fit(X_sc, Y_sc, epochs=EPOCHS, validation_split=0.2, batch_size=1)

# Save model to a file
model.save("summation.keras")
