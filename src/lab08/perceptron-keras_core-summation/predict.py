# Predictions, using previously trained neural network

# Set TensorFlow backend
import os
import joblib

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras_core.models import load_model
import numpy

# Load test dataset
dataset = numpy.loadtxt("test_data.csv", delimiter=",")

# Input vector size
NIN = 2

# Split into input (X) and output (Y) variables
X = dataset[:, 0:NIN]
Y = dataset[:, NIN:]

scaler_X = joblib.load('scaler_X.save')
X_sc = scaler_X.transform(X)
print(X)
print(X_sc)

scaler_Y = joblib.load('scaler_Y.save')

model = load_model("summation.keras")

predictions_sc = model.predict(X_sc)
print(predictions_sc)
predictions = scaler_Y.inverse_transform(predictions_sc)
print(predictions)

print("\nTest results:")
# Caution: the following loop makes sense only for summation of 2 numbers
for i in range(len(Y)):
    print(f"{X[i][0]} + {X[i][1]} = {predictions[i][0]:.4f} (expected: {Y[i][0]})")
