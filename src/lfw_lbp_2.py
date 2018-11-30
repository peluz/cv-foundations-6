import matplotlib.pyplot as plt

import numpy as np
from imutils import paths
import cv2 as cv
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from local_binary_patterns import LocalBinaryPatterns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import random

HEIGHT = 112
WIDTH = 112

def main():
    image_path = sorted(list(paths.list_images("data/images")))
    random.shuffle(image_path)
    data, labels = get_dataset(image_path)
    (X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25)
    

    model = SVC(gamma='auto')
    # model = MLPClassifier(hidden_layer_sizes=(X_train.shape[0],), solver='lbfgs',
                    # activation='tanh',learning_rate_init=0.01)
    model.fit(X_train, Y_train)
    pred = model.predict(X_test)
    # prediction = model.predict(hist.reshape(1, -1))
    print(accuracy_score(Y_test, pred))

def get_dataset(image_path):
    dataset = []
    labels = []
    desc = LocalBinaryPatterns(24, 8)
    for img in image_path:
        image = cv.imread(img)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        dataset.append(hist)
        label = img.split(os.path.sep)[-2]
        labels.append(label)

    dataset = np.array(dataset, dtype="float")
    dataset = normalize(dataset)
    labels = np.array(labels)
    return dataset, labels


if __name__ == "__main__":
    main()