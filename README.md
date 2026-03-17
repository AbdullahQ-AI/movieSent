# 🎬 MovieSent — Dual Approach Sentiment Analysis

AI-powered movie review sentiment analysis using 
Logistic Regression and LSTM Neural Network.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-black)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Accuracy](https://img.shields.io/badge/LR_Accuracy-90.33%25-green)

---

## 🎯 About

Automatically analyzes movie reviews and determines 
whether sentiment is Positive or Negative using two 
different AI approaches.

---

## 🏆 Model Performance

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Logistic Regression | 90.33% ⭐ | 90.40% |
| LSTM Neural Network | 87.53% | 87.51% |

---

## 🔧 Tech Stack

- **Language:** Python 3.10
- **Backend:** Flask
- **ML:** Scikit-learn (Logistic Regression + TF-IDF)
- **DL:** TensorFlow/Keras (LSTM)
- **NLP:** NLTK
- **Dataset:** IMDB 50K Movie Reviews

---

## 📁 Project Structure
```
movieSent/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── models/
│   ├── lr_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── tokenizer_v2.json
│   └── lstm_weights.npy
└── requirements.txt
```

---

## 🚀 How to Run
```bash
git clone https://github.com/AbdullahQ-AI/movieSent.git
cd movieSent
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

---

## 👨‍💻 Author

**Abdullah Qadeer**
GitHub: [@AbdullahQ-AI](https://github.com/AbdullahQ-AI)
```

`Ctrl + S` save karo!

---

## Step 3 — GitHub Pe Repo Banao:
```
1. github.com kholo
2. "+" → New repository
3. Name: movieSent
4. Public
5. Create repository