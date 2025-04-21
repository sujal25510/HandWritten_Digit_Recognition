from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import joblib
import cv2
import numpy as np
from skimage.feature import hog

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model and scaler
clf, scaler = joblib.load('digits_cls.pkl')

# Helpers
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    annotated_file = None
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # Read and preprocess image
            image = cv2.imread(path)
            gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur  = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 90, 255, cv2.THRESH_BINARY_INV)

            # Find contours
            contours, _ = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Annotate
            for ctr in contours:
                x, y, w, h = cv2.boundingRect(ctr)
                length = int(h * 1.6)
                pt1 = max(0, y + h//2 - length//2)
                pt2 = max(0, x + w//2 - length//2)
                roi = thresh[pt1:pt1+length, pt2:pt2+length]
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                kernel = np.ones((3, 3), np.uint8)
                roi = cv2.dilate(roi, kernel, iterations=1)

                fd = hog(
                    roi, orientations=9,
                    pixels_per_cell=(14,14),
                    cells_per_block=(1,1),
                    visualize=False
                )
                fd_scaled = scaler.transform([fd])
                nbr = clf.predict(fd_scaled)[0]

                # Draw box and label
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(
                    image, str(int(nbr)), (x, y-5),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,255,255), 2
                )

            # Save annotated image
            annotated_file = 'annotated_' + filename
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], annotated_file), image)

    return render_template('index.html', annotated=annotated_file)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', debug=True)
