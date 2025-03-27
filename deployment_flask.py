import os
from flask import Flask, request, render_template, jsonify, send_file
from ultralytics import YOLO
import cv2
from io import BytesIO
import numpy as np

app = Flask(__name__)

# 1) Load your custom YOLOv8 model (.pt file, not .pkl!)
model = YOLO("best.pt")  # or "my_custom_model.pt"

# 2) Ensure the upload folder exists
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 3) Save the uploaded image
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # 4) Read and prepare image
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({'error': 'Failed to read image'})
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 5) Run inference (lower conf to 0.1 to catch more boxes)
    results = model.predict(source=img_rgb, conf=0.1)
    boxes = results[0].boxes  # YOLOv8's Boxes object

    # 6) Draw bounding boxes on the original BGR image
    if boxes is None or len(boxes) == 0:
        cv2.putText(img, "No detections found", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for box in boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls] if cls in model.names else f"class_{cls}"

            cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (int(x_min), int(y_min) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # 7) Send the result back as a PNG
    _, img_encoded = cv2.imencode('.PNG', img)
    img_io = BytesIO(img_encoded.tobytes())
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)


