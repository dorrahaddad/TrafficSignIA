from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from keras.models import load_model
from PIL import Image
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Charger le modèle
model = load_model('cnn_model.h5')

# Classes de prédiction
class_names = [
    'Green Light', 'Red Light', 'Speed Limit 10', 'Speed Limit 100',
    'Speed Limit 110', 'Speed Limit 120', 'Speed Limit 20', 'Speed Limit 30',
    'Speed Limit 40', 'Speed Limit 50', 'Speed Limit 60', 'Speed Limit 70',
    'Speed Limit 80', 'Speed Limit 90', 'Stop'
]

# Dossier pour stocker les images téléchargées
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route principale pour afficher le formulaire
@app.route('/')
def index():
    return render_template('index.html')

# Route pour gérer l'upload et prédire
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Sauvegarde de l'image
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Prédiction
            img = Image.open(filepath).resize((64, 64))  # Adapté à la taille de l'entrée du modèle
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            predictions = model.predict(img)
            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            confidence = round(np.max(predictions) * 100, 2)

            return render_template(
                'index.html',
                filename=file.filename,
                predicted_class=predicted_class,
                confidence=confidence
            )

        except Exception as e:
            return jsonify({'error': str(e)}), 500

# Route pour servir les images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)