import json
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage import io, color, transform
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_image(image_path):
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    image = transform.resize(image, (50, 50))
    return image


def load_and_preprocess_images(folder_path, label: int):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = load_and_preprocess_image(image_path)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)


def visualize_results(images, true_labels, predicted_labels):
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i].reshape((50, 50)), cmap='gray')
        plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}", fontsize=8)
        plt.axis('off')
    plt.show()


def main():
    cats_images, cats_labels = load_and_preprocess_images("cats_dogs-1000/images/class-0-cats", label=0)
    dogs_images, dogs_labels = load_and_preprocess_images("cats_dogs-1000/images/class-1-dogs", label=1)

    X_train = np.concatenate((cats_images, dogs_images), axis=0)
    y_train = np.concatenate((cats_labels, dogs_labels))

    test_labels_file = "cats_dogs-1000/images-test/test_labels.txt"
    with open(test_labels_file, 'r') as file:
        test_labels = json.load(file)

    X_test = []
    y_test = []

    for filename, label in test_labels.items():
        image_path = os.path.join("cats_dogs-1000/images-test", filename)

        if os.path.exists(image_path):
            X_test.append(load_and_preprocess_image(image_path))
            y_test.append(1 if label == "dogs" else 0)
    X_test = np.array(X_test)

    model = make_pipeline(StandardScaler(), SVC())

    model.fit(X_train.reshape((X_train.shape[0], -1)), y_train)

    y_pred = model.predict(X_test.reshape((X_test.shape[0], -1)))

    accuracy = accuracy_score(y_test, y_pred)
    print("Validation Accuracy:", accuracy)

    visualize_results(X_test, y_test, y_pred)


if __name__ == '__main__':
    main()
