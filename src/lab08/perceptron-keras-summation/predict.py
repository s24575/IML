# Predictions, using previously trained neural network

from tensorflow.keras.models import load_model
import numpy

# Load test dataset
dataset = numpy.loadtxt("test_data.csv", delimiter=",")

# Input vector size
NIN = 2

# Split into input (X) and output (Y) variables
X = dataset[:, 0:NIN]
Y = dataset[:, NIN:]

model = load_model("summation.keras")

predictions = model.predict(X)

print("\nTest results:")
# Caution: the following loop makes sense only for summation of 2 numbers
for i in range(len(Y)):
    print(f"{X[i][0]} + {X[i][1]} = {predictions[i][0]:.4f} (expected: {Y[i][0]})")
