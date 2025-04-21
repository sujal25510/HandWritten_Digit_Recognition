# generateClassifier.py

#!/usr/bin/env python3

import joblib
import numpy as np
from collections import Counter
from skimage.feature import hog
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

def main():
    # Load MNIST from OpenML
    dataset = fetch_openml('mnist_784', version=1)
    features = np.array(dataset.data, dtype='int16')
    labels   = np.array(dataset.target, dtype='int')

    # Extract HOG features for each image
    list_hog_fd = []
    for img in features:
        fd = hog(
            img.reshape((28, 28)),
            orientations=9,
            pixels_per_cell=(14, 14),
            cells_per_block=(1, 1),
            visualize=False
        )
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, dtype='float64')

    # Standardize features
    scaler = StandardScaler().fit(hog_features)
    hog_features = scaler.transform(hog_features)

    print("Count of digits in dataset:", Counter(labels))

    # Train linear SVM
    clf = LinearSVC(max_iter=10000)
    clf.fit(hog_features, labels)

    # Save classifier + scaler
    joblib.dump((clf, scaler), "digits_cls.pkl", compress=3)
    print("Saved trained model to digits_cls.pkl")

if __name__ == "__main__":
    main()
