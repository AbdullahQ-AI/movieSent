import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout,
                                      Bidirectional, SpatialDropout1D)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

app = Flask(__name__)

print("🔄 Loading models...")

# Logistic Regression + TF-IDF
lr_model = joblib.load('models/lr_model.pkl')
tfidf    = joblib.load('models/tfidf_vectorizer.pkl')

# Tokenizer
with open('models/tokenizer_v2.json', 'r') as f:
    tok_data = json.load(f)

tokenizer = Tokenizer(num_words=tok_data['num_words'],
                      oov_token=tok_data['oov_token'])
tokenizer.word_index = tok_data['word_index']
tokenizer.index_word = {v: k for k, v in tok_data['word_index'].items()}

# LSTM — weights se load karo
def build_lstm():
    model = Sequential([
        Embedding(10000, 64, input_length=200),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

lstm_model = build_lstm()
lstm_model.build(input_shape=(None, 200))
weights = np.load('models/lstm_weights.npy', allow_pickle=True)
lstm_model.set_weights(weights)

print("✅ All models loaded!")

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words -= {'not', 'no', 'never', 'nor', 'neither'}

def preprocess_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(w) for w in tokens
               if w not in stop_words]
    return ' '.join(cleaned)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data   = request.get_json()
        review = data.get('review', '')
        model  = data.get('model', 'lr')

        if not review.strip():
            return jsonify({'error': 'Review cannot be empty!'}), 400

        cleaned = preprocess_text(review)

        if model == 'lr':
            vectorized  = tfidf.transform([cleaned])
            prediction  = lr_model.predict(vectorized)[0]
            probability = lr_model.predict_proba(vectorized)[0]
            confidence  = float(max(probability)) * 100
            model_name  = "Logistic Regression"
        else:
            MAX_LENGTH = 200
            seq        = tokenizer.texts_to_sequences([cleaned])
            padded     = pad_sequences(seq, maxlen=MAX_LENGTH,
                                       padding='post',
                                       truncating='post')
            prob       = lstm_model.predict(padded, verbose=0)[0][0]
            prediction = 1 if prob > 0.5 else 0
            confidence = float(prob if prob > 0.5 else 1-prob) * 100
            model_name = "LSTM"

        sentiment = "Positive" if prediction == 1 else "Negative"

        return jsonify({
            'sentiment' : sentiment,
            'confidence': round(confidence, 2),
            'model'     : model_name,
            'review'    : review[:100] + '...' if len(review) > 100 else review
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)