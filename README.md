## What the Program Does

This program builds a complete NLP pipeline for detecting fraudulent job postings.

The script downloads the Kaggle fake job posting dataset, loads it into a Pandas DataFrame, and combines several text fields into one input column. The combined text includes the job title, company profile, job description, requirements, and benefits.

After preprocessing the data, the program splits it into training and testing sets while preserving the class balance between legitimate and fraudulent postings.

The project then trains and evaluates three different models:

1. **TF-IDF + Logistic Regression**  
   A traditional machine learning baseline that converts job posting text into TF-IDF features and classifies postings with logistic regression.

2. **Fine-Tuned DistilBERT**  
   A transformer-based model that is fine-tuned directly on the job posting dataset for binary classification.

3. **DistilBERT Feature Extraction + Logistic Regression**  
   A hybrid approach where DistilBERT generates text embeddings, and logistic regression is trained on those embeddings.

The program compares the models using accuracy, precision, recall, F1-score, ROC-AUC, classification reports, and confusion matrices.

## Main Goal

The goal of this project is to compare traditional NLP techniques with transformer-based approaches and evaluate how well each method can identify potentially fraudulent job postings.

## Output

When the program runs successfully, it prints model evaluation results to the terminal and saves generated result files such as model comparison charts and confusion matrices.
