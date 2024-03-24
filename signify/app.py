import os
from flask import Flask, render_template, Response, jsonify
from keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Get the directory path of the current script (app.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the relative path to keras_model.h5
MODEL_FILE_PATH = os.path.join(base_dir, "signify/keras_model.h5")

# Load the model
model = load_model(MODEL_FILE_PATH)

optimizer = Adam()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load class names
class_names_file = os.path.join(base_dir, "signify/labels.txt")
class_names = [line.strip() for line in open(class_names_file)]

current_class_name = "Loading..."


def get_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Failed to open camera.")
    return camera


def generate_frames(camera):
    while True:
        ret, frame = camera.read()
        if not ret:
            raise RuntimeError("Failed to read frame from camera.")

        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
        image = (image / 127.5) - 1
        prediction = model.predict(image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        global current_class_name
        if confidence_score > 0.9:
            current_class_name = class_name

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    try:
        camera = get_camera()
        return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
    except RuntimeError as e:
        return str(e), 500


@app.route('/get_class_name')
def get_class_name():
    global current_class_name
    return jsonify({'class_name': current_class_name})

@app.route('/sign')
def sign():
    return render_template('sign.html')

@app.route('/home')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
