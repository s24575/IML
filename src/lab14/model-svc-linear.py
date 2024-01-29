import os
import json

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage import io, transform


def process_image(image):
    image = transform.resize(image, (200, 200))
    image = image.flatten()
    return image


def load_images_and_labels(path):
    images = []
    for filename in os.listdir(path):
        img_path = os.path.join(path, filename)
        if img_path.endswith(".jpg"):
            image = io.imread(img_path)
            image = process_image(image)
            images.append(image)
    return np.array(images)


def main():
    classes = ['cats', 'dogs']

    # Path to the training images
    train_path_cats = 'cats_dogs-1000/images/class-0-cats'
    train_path_dogs = 'cats_dogs-1000/images/class-1-dogs'

    # Path to the test images
    test_path = 'cats_dogs-1000/images-test'

    with open(os.path.join(test_path, 'test_labels.txt'), 'r') as file:
        labels = json.load(file)

    images_cats = load_images_and_labels(train_path_cats)
    labels_cats = [0 for _ in range(len(images_cats))]
    images_dogs = load_images_and_labels(train_path_dogs)
    labels_dogs = [1 for _ in range(len(images_dogs))]

    # Combine cat and dog images and labels
    X = np.concatenate([images_cats, images_dogs], axis=0)
    y = labels_cats + labels_dogs

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM model
    model = SVC()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Predict the test images and display them
    for filename, label in labels.items():
        img_path = os.path.join(test_path, filename)
        if img_path.endswith(".jpg"):
            image = io.imread(img_path)
            processed_image = process_image(image)
            processed_image = processed_image.reshape(1, -1)
            pred_label = model.predict(processed_image)
            output_image = cv2.resize(image, (500, 500))
            image_label = f"original: {label}, predicted: {classes[pred_label[0]]}"
            cv2.putText(output_image, image_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)
            cv2.imshow("Output", output_image)
            key = cv2.waitKey(0)
            if key & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
