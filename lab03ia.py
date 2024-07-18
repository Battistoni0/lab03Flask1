import joblib
import re
import nltk
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

# Variable de texto a analizar
text_to_analyze = "I eat food today"

# Preprocesar el texto
clean_text = preprocess_text(text_to_analyze)

# Transformar el texto con el vectorizador
text_vector = vectorizer.transform([clean_text])

# Predecir el sentimiento
prediction = model.predict(text_vector)

# Mostrar el resultado
sentiment = "Positive" if prediction == 1 else "Negative"
print(f'Text: "{text_to_analyze}" - Sentiment: {sentiment}')
