# Credit Card Fraud Detection

This project detects fraudulent credit card transactions using an Artificial Neural Network (ANN).

## Project Structure
- train_simple.py → Train the ANN model
- predict_simple.py → Make predictions on new data
- requirements.txt → Python dependencies

## How to Run
1. create a new venv:
   python -m venv venv

2. Activate the venv:
   .\venv\Scripts\Activate.ps1

3. Install dependencies:
   pip install -r requirements.txt

4. Run scripts:
   Training: python credit_card_fraud_ann\train_simple.py
   Prediction: python credit_card_fraud_ann\predict_simple.py

   ## Dataset
The original dataset (creditcard.csv, ~140MB) can be downloaded from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).  
For convenience, this repo includes a smaller processed version: creditcard_final.csv.