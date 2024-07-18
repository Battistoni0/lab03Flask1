from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import nltk
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Descargar recursos de NLTK necesarios (ejecutar solo una vez en tu entorno local si no están descargados)
nltk.download('stopwords')
nltk.download('wordnet')

# Función de limpieza de texto
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'@\w+', '', text)  # Eliminar menciones
    text = re.sub(r'#\w+', '', text)  # Eliminar hashtags
    text = re.sub(r'\d+', '', text)  # Eliminar números
    text = re.sub(r'\W', ' ', text)  # Eliminar caracteres especiales
    text = text.lower()  # Convertir a minúsculas
    text = text.split()  # Tokenizar
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

# Cargar el modelo y el vectorizador desde la carpeta models
model = joblib.load('lab03ia/models/sentiment_model.pkl')
vectorizer = joblib.load('lab03ia/models/tfidf_vectorizer.pkl')

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilitar CORS

# Ruta para predecir texto individual
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text_to_analyze = data['text']
    clean_text = preprocess_text(text_to_analyze)
    text_vector = vectorizer.transform([clean_text])
    prediction = model.predict(text_vector)

    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({'text': text_to_analyze, 'sentiment': sentiment})

# Ruta para predecir múltiples textos desde archivos CSV o TXT
@app.route('/upload_files', methods=['POST'])
def upload_files():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    files = request.files.getlist('file')
    if not files:
        return jsonify({'error': 'No files selected'}), 400

    all_texts_to_analyze = []
    for file in files:
        save_path = os.path.join('lab03ia/files', file.filename)
        file.save(save_path)

        if file.filename.endswith('.csv'):
            df = pd.read_csv(save_path)
            texts_to_analyze = df.iloc[:, 0].tolist()  # Supone que los textos están en la primera columna
        elif file.filename.endswith('.txt'):
            with open(save_path, 'r', encoding='utf-8') as f:
                texts_to_analyze = f.readlines()

        all_texts_to_analyze.extend(texts_to_analyze)
        os.remove(save_path)  # Eliminar el archivo después de procesarlo

    clean_texts = [preprocess_text(text) for text in all_texts_to_analyze]
    text_vectors = vectorizer.transform(clean_texts)
    predictions = model.predict(text_vectors)

    sentiments = ["Positive" if prediction == 1 else "Negative" for prediction in predictions]

    return jsonify({'texts': all_texts_to_analyze, 'sentiments': sentiments})

if __name__ == '__main__':
    app.run(debug=True)
