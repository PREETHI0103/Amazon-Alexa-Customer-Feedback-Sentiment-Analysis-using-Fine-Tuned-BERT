# Amazon-Alexa-Customer-Feedback-Sentiment-Analysis-using-Fine-Tuned-BERT
A production-grade AI platform built on customer feedback, leveraging fine-tuned BERT for sentiment intelligence. The system covers data preprocessing, EDA, advanced NLP pipelines, model training and optimization, Hugging Face model hosting, and deployment as a high-performance Streamlit web application for real-time decision support.

# Amazon-Alexa-Customer-Feedback-Sentiment-Analysis-using-Fine-Tuned-BERT

**AI Customer Sentiment Intelligence Platform 📊** — Real-time analysis of Amazon Alexa reviews using **fine-tuned BERT & DistilBERT models** 🚀  

[🔗 Try the live app on Hugging Face Spaces](https://huggingface.co/spaces/PREETHI0103/Customer-Sentiment-Intelligence-Platform)

---

## 📝 Problem Statement / Project Overview

This project aims to convert **unstructured Amazon Alexa customer reviews** into actionable insights using state-of-the-art NLP techniques. The key goals include:

- **Automatic sentiment classification**: Identify whether a review is positive or negative.  
- **End-to-end NLP pipeline**: From **data cleaning**, **exploratory data analysis (EDA)**, and **text preprocessing**, to **model training** and evaluation.  
- **Advanced deep learning models**: Leveraging **DistilBERT** and **BERT**, fine-tuned on Amazon Alexa reviews.  
- **Real-time deployment**: Interactive **Streamlit web application** for instant sentiment predictions.  
- **Model hosting & sharing**: Models hosted on **Hugging Face Hub** for accessibility and reproducibility.  

Dataset: [Amazon Alexa Reviews (Kaggle)](https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews)  
Records: **3,150** | Target column: `feedback` | Classes: `0 = Negative`, `1 = Positive`  

---

## 🔧 Data Preprocessing & Feature Engineering

- Removed **empty rows** and irrelevant entries.  
- Converted text to **lowercase**, removed **punctuation, digits, URLs, mentions, hashtags**.  
- Performed **tokenization**, **spell checking**, **stop word removal**, **lemmatization**, and **TF-IDF vectorization**.  
- Generated **WordClouds** and other visualizations to understand **word frequency and sentiment patterns**.  
- Extensive **EDA** to explore:
  - Distribution of feedback classes  
  - Review length patterns  
  - Impact of ratings on sentiment  
  - Detection of **class imbalance & noise patterns**, etc.  

---

## 🛠️ Tech Stack / Libraries Used

- **Python 3.12** – Programming language  
- **Pandas & NumPy** – Data manipulation & analysis  
- **Matplotlib & Seaborn** – Data visualization  
- **NLTK** – NLP processing (tokenization, stopwords, lemmatization)  
- **SpellChecker** – Text correction  
- **Scikit-learn** – Traditional ML models: Logistic Regression, Random Forest, SVM, XGBoost, Naive Bayes, KNN  
- **Transformers (Hugging Face)** – Fine-tuned **BERT** & **DistilBERT**  
- **Torch / PyTorch** – Deep learning framework for training & inference  
- **Streamlit** – Web deployment for real-time sentiment analysis  
- **Hugging Face Hub** – Model & app hosting  
- **WordCloud** – Text pattern visualization  

---

## 🚀 Machine Learning & Deep Learning Models

- Traditional ML models were trained for baseline comparison: **Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, Naive Bayes, KNN**  
- Advanced transformer models:
  - **DistilBERT**: Lightweight BERT variant for faster inference.  
  - **BERT**: Full transformer model, fine-tuned for highest performance.  

---

## 📊 Model Evaluation

**DistilBERT (Fine-Tuned):**  
- Accuracy: 0.9587  
- Balanced Accuracy: 0.8166  
- F1-Score: 0.9777  
- Confusion Matrix:

