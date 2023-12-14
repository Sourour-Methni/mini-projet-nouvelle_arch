from flask import Flask, request, jsonify
import joblib
import base64
import io
import numpy as np
import librosa
import pickle

app = Flask(__name__)
# Load the trained SVM model and PCA instance
svm_model = joblib.load('svm_model.pkl')
with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)
@app.route('/svm', methods=['POST'])
def predict_genre_svm():
    try:
        audio_file = request.files['audio_data']
        if audio_file is None:
            return jsonify({'error': 'No audio file received'})
        # Read the content of the audio file
        audio_content = audio_file.read()
        # Decode the audio data (if encoded, such as in base64)
        audio_bytes = base64.b64decode(audio_content)  # Make sure to use the appropriate decoding method if necessary
        # Convert the audio data into a numpy array (floating-point)
        audio, sample_rate = librosa.load(io.BytesIO(audio_bytes), sr=None)
        # Example: Feature extraction using MFCC (adjust as needed)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=58)

        # Apply PCA transformation using the previously fitted PCA instance with 21 components
        transformed_features = pca.transform(mfcc.T)  # Transpose the MFCC features if needed

        # Make predictions with the trained SVM model
        predictions = svm_model.predict(transformed_features)
        # Map predicted genre labels to their names
        genre_names = {
            1: 'classical',
            2: 'blues',
            3: 'country',
            4: 'disco',
            5: 'hiphop',
            6: 'jazz',
            7: 'metal',
            8: 'pop',
            9: 'reggae',
            10: 'rock'
        }
        # Select the most common predicted genre and get its name
        predicted_genre_label = int(np.bincount(predictions).argmax())
        predicted_genre_name = genre_names.get(predicted_genre_label, 'Unknown')

        return jsonify({'predicted_genre': predicted_genre_name})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=510)
