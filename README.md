# 🎬 IMDB Movie Review Sentiment Analysis

## 📖 Overview  
This project analyzes and predicts the **sentiment** of IMDB (Internet Movie Database) movie reviews using **Natural Language Processing (NLP)** and **Machine Learning**.  
The aim is to automatically classify whether a review is **positive** or **negative**, helping platforms understand audience opinions at scale.

---

## 🎯 Objective  
To build and evaluate models that can accurately determine the sentiment of a movie review text (positive or negative).

---

## 📊 Dataset Description  
The dataset contains **50,000 IMDB movie reviews**, each labeled as either *positive* or *negative*.

| Column | Description |
|---------|-------------|
| `review` | The text content of the movie review |
| `sentiment` | Sentiment label — **positive** or **negative** |

---

## ⚙️ Data Preprocessing & Cleaning  
The text data was prepared using several preprocessing steps:  
- Removed HTML tags, special characters, and punctuation  
- Converted all text to lowercase  
- Removed stopwords and performed stemming/lemmatization  
- Tokenized text into individual words  
- Transformed reviews into numerical vectors using **TF-IDF**  
- Split dataset into **training** and **testing** sets  

---

## 🤖 Model Development  
Several models were trained and compared to evaluate performance on sentiment classification:

### 🧩 Machine Learning Models  
- Logistic Regression  
- Support Vector Machine (SVM)  
- Random Forest  
- Naive Bayes  

### ⚙️ Tuned Models  
- Logistic Regression (tuned)  
- Linear SVC (tuned)  

### 🧠 Advanced Deep Learning Model  
- **DistilBERT (fine-tuned)** — a transformer-based NLP model for contextual sentiment understanding.

---

## 📈 Model Evaluation  
All models were evaluated using key metrics:  
- **Accuracy**  
- **Precision**  
- **Recall**  
- **F1 Score**

| Model | Description | F1 Score |
|--------|--------------|----------|
| Logistic Regression (base) | Baseline linear model using TF-IDF | **0.897** |
| SVM (base) | Support Vector Machine classifier | **0.892** |
| Random Forest (base) | Ensemble model with TF-IDF | 0.863 |
| Naive Bayes (base) | Probabilistic text classifier | 0.860 |
| Logistic Regression (tuned) | Optimized hyperparameters | 0.886 |
| Linear SVC (tuned) | Fine-tuned SVM variant | 0.857 |
| DistilBERT (fine-tuned) | Transformer-based deep learning model | 0.743 |

---

### 📊 Model Comparison (Test F1)
![Model Comparison](4b124712-7b7d-4516-a846-19c912b90a0e.png)

*(Visualization comparing F1 scores of different models)*

---

### 🔍 Observations  
- Traditional ML models like **Logistic Regression** and **SVM** outperformed other approaches on TF-IDF features.  
- **DistilBERT**, though advanced, required more fine-tuning to achieve better performance.  
- **Logistic Regression** achieved the best F1 score (0.897) with faster training and good interpretability.  

---

## 🧠 Key Insights  
- The dataset is balanced, providing equal representation of positive and negative reviews.  
- Text preprocessing (especially TF-IDF transformation) greatly improved model performance.  
- Logistic Regression and SVM are strong baselines for text classification tasks.  
- Transformer-based models like DistilBERT can perform better with more computational power and fine-tuning.

---

## 🧰 Tools & Technologies Used  
- **Python**  
- **Pandas**, **NumPy**  
- **Matplotlib**, **Seaborn**  
- **NLTK**, **Scikit-learn**, **Hugging Face Transformers**  
- **Jupyter Notebook**

---


