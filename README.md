# YouTube Comment Sentiment Analysis Extension

## Project Overview
This project is a **Google Chrome extension** that analyzes YouTube comments and performs **sentiment analysis** to classify them as **Positive**, **Neutral**, or **Negative**.  

The workflow is divided into two main steps:

---

## Step 1: Data Analysis & Model Development
- **Dataset:** Combined Reddit and Twitter comments labeled with sentiment (`-1`, `0`, `1`).  
- **Exploratory Data Analysis (EDA):** Cleaned and preprocessed text data, analyzed patterns and distributions.  
- **Feature Engineering & NLP:**  
  - Tokenization: TF-IDF vs Bag-of-Words (BoW)  
  - Data balancing techniques applied to improve model performance.  
- **Models Tested:** SVM, XGBoost, Random Forest, LightGBM  
- **Hyperparameter Tuning:** Performed for all models using **MLflow experiments**.  
- **Model Selection:** Best combination of model, tokenizer, and data balancer selected based on evaluation metrics.  
- **Experiment Tracking:**  
  - MLflow used to log experiments  
  - Hosted on **Google Cloud GCE VM** with **Google Cloud Storage bucket** for artifacts.  

---

## Step 2: Deployment & Extension
- **Pipeline & Versioning:**  
  - DVC used to manage data and pipeline stages:  
    1. Data Preprocessing  
    2. Data Ingestion  
    3. Model Building  
    4. Model Evaluation  
    5. Model Registration  
- **Backend API:**  
  - FastAPI used to serve the model  
  - Exposes endpoints to predict sentiment for new comments.  
- **Google Chrome Extension:**  
  - Built using **HTML, CSS, JSON** to interact with the FastAPI backend.  
  - Fetches YouTube comments, sends them to the API, and displays sentiment analysis results.  
- **Containerization:** Dockerized the entire project.  
- **Hosting:**  
  - Deployed using **Google Compute Engine (GCE)**  
  - **CI/CD** implemented via **GitHub Actions** (workflow YAML included)  
  - DVC pipeline YAML also included for reproducibility  

---

## Features
- Fetches and analyzes YouTube comments in real-time.  
- Provides sentiment classification (**Positive**, **Neutral**, **Negative**).  
- Visualizations:
  - Sentiment distribution chart  
  - Trend over time graph  
  - Word cloud of comments  

---

## Tech Stack
- **Python:** FastAPI, pandas, numpy, scikit-learn, LightGBM, NLTK, Matplotlib, WordCloud  
- **MLOps:** MLflow, DVC, Google Cloud Storage, GCE  
- **Frontend:** Chrome Extension (HTML, CSS, JS)  
- **Deployment:** Docker, GitHub Actions (CI/CD)  

---

## Usage
1. **Clone the repository:**  
   ```bash
   git clone https://github.com/USERNAME/MLops_Sentiments_classification.git
   cd MLops_Sentiments_classification
