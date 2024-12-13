from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

cnn_model_1 = load_model('cnn_traffic_sign_model_4.h5')
cnn_model_2 = load_model('cnn_traffic_sign_model_5.h5')

yolo_model_1 = YOLO('best_yolo1.pt')
yolo_model_2 = YOLO('best_yolo2.pt')

class_names = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]

UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    selected_model = request.form.get('model')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Save the image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Call appropriate prediction function based on the selected model
            if selected_model == 'cnn1':
                return cnn_predict(filepath, cnn_model_1, 'CNN Model 1', (222, 222))
            elif selected_model == 'cnn2':
                return cnn_predict(filepath, cnn_model_2, 'CNN Model 2', (414, 414))
            elif selected_model == 'yolo1':
                return yolo_predict(filepath, yolo_model_1, 'YOLO Model 1')
            elif selected_model == 'yolo2':
                return yolo_predict(filepath, yolo_model_2, 'YOLO Model 2')

        except Exception as e:
            return jsonify({'error': str(e)}), 500


def cnn_predict(filepath, model, model_name, input_size):
    # Preprocess the image
    img = Image.open(filepath).resize(input_size)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make predictions
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    confidence = round(np.max(predictions) * 100, 2)

    return render_template(
        'index.html',
        filename=os.path.basename(filepath),
        predicted_class=predicted_class,
        confidence=confidence,
        model_used=model_name
    )


def yolo_predict(filepath, model, model_name):
 
    image = cv2.imread(filepath)
    results = model.predict(image)

    annotated_image = results[0].plot()
    annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], 'annotated_' + os.path.basename(filepath))
    cv2.imwrite(annotated_path, annotated_image)

    predictions = results[0].boxes.data.cpu().numpy()
    detected_objects = [
        {
            "name": results[0].names[int(box[5])],
            "confidence": float(box[4])
        }
        for box in predictions
    ]

    return render_template(
        'index.html',
        filename='annotated_' + os.path.basename(filepath),
        detected_objects=detected_objects,
        model_used=model_name
    )


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/model_details/<model_name>', methods=['GET'])
def model_details(model_name):
    try:
        if model_name == 'cnn1':
            confusion_matrix_image = 'static/model_details/cnn2_confusion_matrix.png'
            accuracy = '42%'
        elif model_name == 'cnn2':
            confusion_matrix_image = 'static/model_details/cnn6_confusion_matrix.png'
            accuracy = '8.08%'
        elif model_name == 'yolo1':
            confusion_matrix_image = 'static/model_details/yolo1_confusion_matrix.png'
            accuracy = '79.36%'
        elif model_name == 'yolo2':
            confusion_matrix_image = 'static/model_details/yolo2_confusion_matrix.png'
            accuracy = '76.52%'
        else:
            return jsonify({'error': 'Invalid model selected'}), 400

        return jsonify({
            'accuracy': accuracy,
            'confusion_matrix_image': f"/{confusion_matrix_image}"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
