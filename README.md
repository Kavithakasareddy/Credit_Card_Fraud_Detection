# 💳 Credit Card Fraud Detection

A machine learning web application that detects whether a credit card transaction is normal or fraudulent.

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Logistic Regression

## Features

- Loads credit card transaction data
- Trains a Logistic Regression model
- Displays model accuracy
- Allows users to enter transaction details
- Predicts whether a transaction is normal or fraudulent
- Displays fraud probability

## Dataset

The Credit Card Fraud Detection dataset is not included in this repository because of its large file size.

The application expects the dataset at:

dataset/creditcard.csv

## How to Run

Install the required libraries:

pip install -r requirements.txt

Run the application:

streamlit run app.py
