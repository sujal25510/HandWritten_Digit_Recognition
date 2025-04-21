# performRecognition.py

#!/usr/bin/env python3

import argparse
import cv2
import joblib
import numpy as np
from skimage.feature import hog

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--classifierPath",
        help="Path to classifier file (digits_cls.pkl)",
        required=True
    )
    parser.add_argument(
        "-i", "--image",
        help="Path to input image",
        required=True
    )
    args = parser.parse_args()

    # Load trained model and scaler
    clf, scaler = joblib.load(args.classifierPath)

    # Read and preprocess image
    image = cv2.imread(args.image)
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours (OpenCV ≥4 returns 2 values)
    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        # Draw rectangle around digit
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract ROI, resize to 28x28
        length = int(h * 1.6)
        pt1 = max(0, y + h//2 - length//2)
        pt2 = max(0, x + w//2 - length//2)
        roi = thresh[pt1:pt1+length, pt2:pt2+length]
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

        # Dilate to close gaps (3×3 kernel)
        kernel = np.ones((3, 3), np.uint8)
        roi = cv2.dilate(roi, kernel, iterations=1)

        # Compute HOG descriptor
        fd = hog(
            roi,
            orientations=9,
            pixels_per_cell=(14, 14),
            cells_per_block=(1, 1),
            visualize=False
        )
        fd_scaled = scaler.transform([fd])

        # Predict and annotate
        nbr = clf.predict(fd_scaled)[0]
        cv2.putText(
            image,
            str(int(nbr)),
            (x, y - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            1.5,
            (0, 255, 255),
            2
        )

    # Display result
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
