from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model('vgg.h5')

genreMap = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

@app.route('/predict', methods=['POST'])
def predict_genre():
    data = request.json
    image = data['image']  # Suppose you pass the image data in the request

    # Pré-traitement de l'image (adapter selon vos besoins)
    processed_image = preprocess_image(image)

    # Faire la prédiction avec le modèle
    predictions = model.predict(processed_image)
    predicted_genre = genreMap[np.argmax(predictions)]

    return jsonify({'predicted_genre': predicted_genre})

def preprocess_image(image):
    # Redimensionner l'image à la taille attendue par le modèle
    resized_image = image.resize((288, 432))  # Adapter à la taille attendue par votre modèle

    # Normaliser les pixels de l'image entre 0 et 1
    processed_image = np.array(resized_image) / 255.0

    # Adapter la forme pour correspondre à celle attendue par le modèle (si nécessaire)
    processed_image = np.expand_dims(processed_image, axis=0)  # Si le modèle attend un lot (batch)

    return processed_image

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=800)
