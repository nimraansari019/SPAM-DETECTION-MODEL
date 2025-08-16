# SPAM-DETECTION-MODEL



# Spam Email Detection with Machine Learning

## 🔹 Introduction

Email has become one of the most important forms of digital communication, but it also carries the risk of spam messages. Spam emails often include advertisements, phishing attempts, and fraudulent links that may harm users. Manually filtering these emails is almost impossible due to the large volume. Hence, machine learning techniques are widely used to **automatically classify emails as “spam” or “ham” (non-spam)**.

In this project, we have built a **Spam Email Detection Model** using **Python, Scikit-learn, and Natural Language Processing (NLP)** techniques. The project demonstrates how raw email text can be cleaned, transformed into numerical features, and then classified using machine learning algorithms.

---

## 🔹 Objectives

1. Load and explore the dataset (SMS Spam Collection dataset).
2. Preprocess the text messages (cleaning, stopword removal, stemming).
3. Convert textual data into numerical features using **TF-IDF vectorization**.
4. Train machine learning models like **Naive Bayes** and **Logistic Regression**.
5. Evaluate model performance using accuracy, confusion matrix, ROC curves, and precision-recall analysis.
6. Test the trained model on custom messages to check predictions.

---

## 🔹 Tools & Libraries Used

* **Programming Language**: Python 
* **Libraries**:

  * `pandas`, `numpy` → data handling
  * `matplotlib`, `seaborn` → data visualization
  * `scikit-learn` → ML algorithms & evaluation
  * `nltk` → natural language preprocessing
* **Editor/Platform**: Jupyter Notebook 

---

## 🔹 Methodology

### 1. Data Loading

We used the **SMS Spam Collection dataset** (UCI repository). The dataset contains messages labeled as either *ham* (legitimate) or *spam*.

### 2. Data Preprocessing

To make the text data machine-readable, preprocessing steps were applied:

* Lowercasing text
* Removing punctuation, numbers, and special characters
* Tokenizing into words
* Removing stopwords (like *the, is, an*)
* Applying stemming (converting words like *running → run*)

This ensured that the dataset was clean and consistent.

### 3. Feature Engineering

Text data was transformed into numerical vectors using **TF-IDF (Term Frequency – Inverse Document Frequency)**. This helps in identifying important words in messages while reducing the weight of commonly used words.

### 4. Model Building

We trained two classification models:

* **Multinomial Naive Bayes** – works well with text classification problems.
* **Logistic Regression** – a linear model for binary classification.

### 5. Model Evaluation

The models were evaluated using:

* **Accuracy Score** (\~97%)
* **Confusion Matrix** (shows correctly & incorrectly classified messages)
* **ROC Curves and AUC Scores** (measure true positive vs false positive rate)
* **Precision-Recall Curves** (useful for imbalanced datasets like spam detection)

Logistic Regression performed slightly better than Naive Bayes in our evaluation.

### 6. Prediction Demonstration

We tested the model with sample messages like:

* “Congratulations! You’ve won \$1000 gift card.” → **Spam**
* “Hey, are we still on for dinner tonight?” → **Ham**

The model correctly predicted the class with high probability scores.

---
##OUTPUT:
<img width="1264" height="505" alt="Image" src="https://github.com/user-attachments/assets/c8221891-dafb-4fd1-b0ee-1977b3635d65" />



## 🔹 Applications

* Email service providers (like Gmail, Yahoo) use similar models to filter spam.
* Can be extended to **phishing detection** and **SMS fraud detection**.
* Useful in **cybersecurity** to protect users from malicious links.

---

## 🔹 Conclusion

This project successfully demonstrates how **machine learning and NLP** can be used to classify emails as spam or ham. By applying preprocessing, TF-IDF vectorization, and ML models, we achieved a **high accuracy of \~97%**. Logistic Regression proved to be the best-performing model in our tests.

Future improvements could include trying **deep learning models (LSTMs, Transformers)** and experimenting with **larger datasets** for even better accuracy.

---

