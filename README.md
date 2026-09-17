# 🛡️ Deep Learning for Comment Toxicity Detection

## 📌 Project Overview

This project develops a Deep Learning based Natural Language Processing system for detecting toxic comments.

The application uses an LSTM-based neural network and provides an interactive Streamlit interface for real-time comment toxicity detection.

The system can also process multiple comments through CSV upload.

---

## 🎯 Objective

The objective of this project is to develop an automated toxicity detection system that can identify harmful or inappropriate comments.

The model predicts the following categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

---

## 💼 Business Use Cases

This solution can be used for:

- Social media content moderation
- Online forums
- Community websites
- E-learning platforms
- News websites
- Brand safety
- Content moderation services

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- NLP
- LSTM
- Streamlit

---

## 📂 Dataset

The dataset contains comments and six toxicity labels.

Columns:

- id
- comment_text
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Text Cleaning
   ↓
Train/Test Split
   ↓
Tokenization
   ↓
Padding
   ↓
LSTM Deep Learning Model
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Save Model
   ↓
Streamlit Application
   ↓
Single Comment Prediction
   ↓
CSV Bulk Prediction
