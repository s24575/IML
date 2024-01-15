import keras
import keras_tuner
from keras.src.metrics import Precision, Recall, F1Score
from matplotlib import pyplot as plt

MODEL_FILENAME = "model.keras"
NO_OF_CLASSES = 10
VAL_SPLIT = 0.2
EPOCHS = 8
BATCH_SIZE = 32


def build_model(hp):
    # Initialize the model
    model = keras.models.Sequential()
    # Flatten the input data of (x, y, 1) dimension
    model.add(keras.layers.Input(shape=(28, 28, 1)))
    model.add(keras.layers.Flatten())

    # FC/RELU layer
    hp_units = hp.Int('units', min_value=4, max_value=128, step=4)
    activation_1 = hp.Choice('activation_1', values=['relu', 'sigmoid'])
    model.add(keras.layers.Dense(units=hp_units, activation=activation_1))

    # Softmax classifier (10 classes)
    model.add(keras.layers.Dense(NO_OF_CLASSES, activation="softmax"))

    optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy', Precision(), Recall(), F1Score()])

    return model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert from uint8 to float32 and normalize to [0,1]
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    y_train = keras.utils.to_categorical(y_train, NO_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NO_OF_CLASSES)

    tuner = keras_tuner.RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=3,
        executions_per_trial=3,
        directory='tuner_logs',
        project_name='mnist_tuning'
    )

    tuner.search_space_summary()
    tuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    tuner.results_summary()

    best_model = tuner.get_best_models()[0]

    history = best_model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(range(EPOCHS), history.history["loss"], label="Train loss")
    plt.plot(range(EPOCHS), history.history["val_loss"], label="Validation loss")
    plt.plot(range(EPOCHS), history.history["accuracy"], label="Train accuracy")
    plt.plot(range(EPOCHS), history.history["val_accuracy"], label="Validation accuracy")

    plt.title("Train / validation loss and accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="right")
    plt.savefig("model.png")

    # Save model to a file
    best_model.save(MODEL_FILENAME)

    # Evaluate the model on the test data
    evaluation = best_model.evaluate(x_test, y_test)
