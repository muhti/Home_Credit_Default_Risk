# Home Credit Risk Assessment Using Machine Learning

## Project Overview

This project develops a model to classify loan applicants at Home Credit who are likely to encounter loan payment difficulties by estimating the probability of default. 

## Objective

The objective is to accurately as possible predict loan defaults, using several machine learning models and deep learning techniques maximizing the ROC-AUC score. 
The model is designed to provide insights for the risk when it come to borrowers.

## Data

application_{train|test}.csv This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET). Static data for all applications. One row represents one loan in our data sample.

bureau.csv All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample). For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

bureau_balance.csv Monthly balances of previous credits in Credit Bureau. This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

POS_CASH_balance.csv Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

credit_card_balance.csv Monthly balance snapshots of previous credit cards that the applicant has with Home Credit. This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

previous_application.csv All previous applications for Home Credit loans of clients who have loans in our sample. There is one row for each previous application related to loans in our data sample.

installments_payments.csv Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample. There is a) one row for every payment that was made plus b) one row each for missed payment. One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

HomeCredit_columns_description.csv This file contains descriptions for the columns in the various data files.

Missing data explanation: XNA - Not available, XAP - Not applicable, 365243 denotes infinity in DAYS variables in the datasets, therefore the can be considered missing values.

## Data aggregations and feature engineering:

Derived features like credit-to-income ratio, income-per-family ratio, annuity-to-income ratio, payment delays, etc.
Aggregations based on previous applications, bureau data, credit card balances, installment payments, and point-of-sale loans.
Log transformations and scaling for handling skewed data and improving model performance.
See preprocessing.py

## Key Results

LightGBM: Achieved a final ROC-AUC score of 0.78 on the test set.
Ensemble model: Improved test ROC-AUC to 0.782 after combining both LightGBM and MLP.
Decision threshold optimization helped in improving the F1 score to 0.33.

## Installation

First, clone the GitHub repository containing the project files to your local machine:

### 1. Clone the Repository

```bash
git clone https://github.com/muhti/home-credit-default-risk.git
cd home-credit-default-risk
```

### 2. Set up a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate   # For Windows: venv\Scripts\activate
```

### 3.Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Flask app

```bash
python app.py
```

### Making Predictions

To use the model, you need to upload a ZIP file containing these datasets:

application_train.csv
bureau.csv
bureau_balance.csv
previous_application.csv
pos_cash_balance.csv
credit_card_balance.csv
installments_payments.csv

### License

[MIT](https://choosealicense.com/licenses/mit/)
