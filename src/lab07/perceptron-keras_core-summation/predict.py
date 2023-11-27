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

scaler = joblib.load('scaler.save')
X = scaler.transform(X)

model = load_model("summation.keras")

predictions = model.predict(X)
print(predictions)
predictions = scaler.inverse_transform(predictions).reshape(-1, 1)
print(predictions)

print("\nTest results:")
# Caution: the following loop makes sense only for summation of 2 numbers
for i in range(len(Y)):
    print(f"{X[i][0]} + {X[i][1]} = {predictions[i][0]:.4f} (expected: {Y[i][0]})")
