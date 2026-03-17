# 🎬 MovieSent — Dual Approach Sentiment Analysis

AI-powered movie review sentiment analysis using Logistic Regression and LSTM Neural Network.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![LR Accuracy](https://img.shields.io/badge/LR_Accuracy-90.33%25-green)
![LSTM Accuracy](https://img.shields.io/badge/LSTM_Accuracy-87.53%25-yellow)

---

## 🎯 About

MovieSent automatically analyzes movie reviews and determines whether the sentiment is Positive or Negative using two different AI approaches — a classical ML model and a Deep Learning model.

---

## 🏆 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 90.33% ⭐ | 89.72% | 91.09% | 90.40% |
| LSTM Neural Network | 87.53% | 87.68% | 87.33% | 87.51% |

---

## 🔧 Tech Stack

- **Language:** Python 3.10
- **Backend:** Flask
- **ML Model:** Scikit-learn (Logistic Regression + TF-IDF N-Grams)
- **DL Model:** TensorFlow/Keras (Bidirectional LSTM)
- **NLP:** NLTK (Tokenization, Stopwords, Lemmatization)
- **Dataset:** IMDB 50K Movie Reviews

---

## ✨ Features

- Dual model selection (Logistic Regression or LSTM)
- Real-time sentiment prediction
- Confidence score display
- Sample reviews to test
- Model performance comparison table
- Responsive dark theme UI

---

## 📁 Project Structure
```
movieSent/
├── app.py                 ← Flask backend
├── templates/
│   └── index.html         ← Frontend UI
├── static/
│   └── style.css          ← Styling
├── models/
│   ├── lr_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer_v2.json
│   └── lstm_weights.npy
└── requirements.txt
```

---

## 🚀 How to Run Locally
```bash
git clone https://github.com/AbdullahQ-AI/movieSent.git
cd movieSent
pip install -r requirements.txt
python app.py
```

Open browser: `http://127.0.0.1:5000`

---

## 👨‍💻 Author

**Abdullah Qadeer**
GitHub: [@AbdullahQ-AI](https://github.com/AbdullahQ-AI)

---

## 📄 License

This project is open source and available under the MIT License.