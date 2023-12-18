import os
# Disable TF warning messages and set backend
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_core as keras
import numpy as np
import cv2     # python -m pip install opencv-python

from keras_core.datasets import mnist

# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model filename
MODEL = 'model.keras'

if __name__ == "__main__":
    
    # Load trained model
    model = keras.models.load_model(MODEL)

    (_, _), (x_test, y_test) = mnist.load_data()

    for image, true_label in zip(x_test, y_test):
        # Pre-process the image for classification
        image_data = image.astype('float32') / 255
        image_data = keras.preprocessing.image.img_to_array(image_data)
        # Expand dimension (28,28,1) -> (1,28,28,1)
        image_data = np.expand_dims(image_data, axis=0)
        
        # Classify the input image
        prediction = model.predict(image_data, verbose=0)
        pred_flat = prediction.flatten()
        
        n = 3
        sorted_indices = np.argsort(pred_flat)[::-1]
        top_n_indices = sorted_indices[:n]
        top_n_probabilities = pred_flat[top_n_indices]

        label_list = []
        for index, prob in zip(top_n_indices, top_n_probabilities):
            prob_pct = prob * 100
            if prob_pct < 1:
                break
            label_list.append("{}: {:0.2f}%".format(index, prob_pct))
        label_pred = ", ".join(label_list)

        label_actual = f"actual: {true_label}"

        # Draw the label on the image
        output_image = cv2.resize(image, (500, 500))
        cv2.putText(output_image, label_pred, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(output_image, label_actual, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Show the output image        
        cv2.imshow("Output", output_image)
        
        # Break on 'q' pressed, continue on the other key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


# Zadanie 9.2 (1p)
# Uzupełnić wyświetlany tekst na obrazie o klasy z miejsca 2 i 3, 
# o ile ich prawdopodobieństwo jest większe od 1%.

# Zadanie 9.3 (1p)
# Zamiast wczytywać obrazy testowe z plików, ładować je metodą mnist.load_data() 
# z API KerasCore.

# Wynik: plik z uzupełnionym kodem oraz plik graficzny 
# z przykładowym wynikiem predykcji (z co najmniej dwiema klasami).